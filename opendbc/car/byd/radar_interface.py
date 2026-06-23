#!/usr/bin/env python3
from opendbc.car.interfaces import RadarInterfaceBase
from opendbc.car import Bus, structs
from opendbc.can.parser import CANParser
from opendbc.car.byd.values import DBC, CanBus

import os
BYD_RADAR = os.getenv("BYD_RADAR") is not None

class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP, CP_SP):
    super().__init__(CP, CP_SP)

    # Kalman filter: state = [pos, vel], measure = pos
    self._kf_pos = 0.0
    self._kf_vel = 0.0
    self._kf_P = [[1.0, 0.0], [0.0, 10.0]]  # covar
    self._kf_init = False

    if CP.radarUnavailable:
      self.rcp = None
    else:
      messages = [('RADAR_MRR', 60)]
      self.rcp = CANParser(DBC[CP.carFingerprint][Bus.pt], messages, CanBus.MPC)
      self.trigger_msg = 0x374

    self.updated_messages = set()

  def _kf_predict(self, dt):
    # x = F * x
    self._kf_pos += self._kf_vel * dt
    # P = F * P * F^T + Q
    dt2 = dt * dt
    self._kf_P[0][0] += dt * (2.0 * self._kf_P[1][0] + dt * self._kf_P[1][1]) + 0.1 * dt2
    self._kf_P[0][1] += dt * self._kf_P[1][1]
    self._kf_P[1][0] += dt * self._kf_P[1][1]
    self._kf_P[1][1] += 1.0 * dt  # process noise for velocity

  def _kf_update(self, z, R=1.0):
    # K = P * H^T * (H * P * H^T + R)^-1
    innov = z - self._kf_pos
    S = self._kf_P[0][0] + R
    K0 = self._kf_P[0][0] / S
    K1 = self._kf_P[1][0] / S
    # x = x + K * innov
    self._kf_pos += K0 * innov
    self._kf_vel += K1 * innov
    # P = (I - K*H) * P
    self._kf_P[0][0] -= K0 * self._kf_P[0][0]
    self._kf_P[0][1] -= K0 * self._kf_P[0][1]
    self._kf_P[1][0] -= K1 * self._kf_P[0][0]
    self._kf_P[1][1] -= K1 * self._kf_P[0][1]

  def update(self, can_strings):
    if self.rcp is None:
      return super().update(None)

    values = self.rcp.update_strings(can_strings)
    self.updated_messages.update(values)

    if self.trigger_msg not in self.updated_messages:
      return None

    ret = structs.RadarData()

    if not self.rcp.can_valid:
      ret.errors.canError = True

    msg_mrr = self.rcp.vl['RADAR_MRR']
    msg_id = msg_mrr['TargetID']
    longdist = msg_mrr['LongDist']
    latdist  = msg_mrr['LatDist']
    isvalid_raw = bool(msg_mrr['IsValid'])

    if msg_id == 2:
      isvalid = longdist > 0 or isvalid_raw

      if msg_id not in self.pts:
        self.pts[msg_id] = structs.RadarData.RadarPoint()
        self.pts[msg_id].trackId = msg_id

      if isvalid:
        # --- KF predict (dt from 60Hz radar) ---
        if self._kf_init:
          ddt = 1.0 / 60.0
          self._kf_predict(ddt)

        # --- KF update with measurement ---
        if self._kf_init:
          self._kf_update(float(longdist), R=1.0)
        else:
          self._kf_pos = float(longdist)
          self._kf_vel = 0.0
          self._kf_P = [[1.0, 0.0], [0.0, 10.0]]
          self._kf_init = True

        self.pts[msg_id].dRel = self._kf_pos
        self.pts[msg_id].yRel = latdist
        self.pts[msg_id].vRel = self._kf_vel
        self.pts[msg_id].vLead = self._kf_vel + self.v_ego
      else:
        self.pts[msg_id].dRel = 255.0
        self.pts[msg_id].yRel = 0.0
        self.pts[msg_id].vRel = float('nan')
        self.pts[msg_id].vLead = float('nan')

      self.pts[msg_id].aRel = float('nan')
      self.pts[msg_id].yvRel = float('nan')
      self.pts[msg_id].measured = isvalid

    ret.points = list(self.pts.values())
    self.updated_messages.clear()
    return ret
