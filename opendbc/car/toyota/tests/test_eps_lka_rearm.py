"""EPS LKA authority loss and recovery (sunnypilot-pc).

Evidence: one real drive (25 segments, 139,038 frames, 24.3 min, Lexus ES TSS2, lateral demanded by
MADS with openpilot disabled). EPS_STATUS->LKA_STATE reported "active" (5) during only 76.4% of the
latActive time: 298.7 s in 294 episodes were spent demanding torque with a torque command on the
wire while the EPS applied far less of it (median |EPS torque| / |command| was 0.49 in standby and
0.53 in the temporary-fault states, 33% of those frames applying under a quarter, against
1.23 when active). The longest episode lasted 11.4 s at 12.6-17.2 m/s with the wheel moving from -4.8 to
+23.7 degrees and STEER_REQUEST high on 1137 of its 1139 frames. Across the drive, 124 of the 153
returns to active were preceded by a request falling edge within 1.2 s, and 416 of 493 rising edges
were followed by the active state within 2 s (median 20 ms). Holding the request high armed nothing.

The two corrections that follow, both under test here:
  * carstate: LKA_STATE 9/21/25 are only reported as temporary steering faults once they persist
    (157 runs lasting 1.25 s median / 1.94 s at the longest, so frame-true reporting meant 157
    warnings in 24 minutes while never reporting them hides an EPS that stays faulted)
  * carcontroller: while torque is demanded and the EPS sits in a re-armable state, drop torque
    and the request for a short window so the EPS gets the low->high edge it requires
"""

import unittest

from opendbc.car import Bus, structs
from opendbc.can.packer import CANPacker
from opendbc.can.parser import CANParser
from opendbc.car.toyota import carstate
from opendbc.car.toyota.carstate import CarState as ToyotaCarState
from opendbc.car.toyota.carcontroller import (CarController as ToyotaCarController, LKA_STATE_ACTIVE,
                                              LKA_STATE_STANDBY, LKA_STATE_UNAVAILABLE, LKA_STATE_FAULT_TEMP,
                                              LKA_STATE_FAULT_TEMP2, LKA_STATE_FAULT_TEMP3, LKA_STATE_PERM_FAULT,
                                              LKA_STATE_REARM)
from opendbc.car.toyota.interface import CarInterface as ToyotaCarInterface
from opendbc.car.toyota.values import CAR, DBC

GearShifter = structs.CarState.GearShifter
LongControlState = structs.CarControl.Actuators.LongControlState

TORQUE_CAR = CAR.LEXUS_ES_TSS2
ANGLE_CAR = CAR.TOYOTA_RAV4_TSS2_2023
FRAME_NANOS = int(1e9 / 100)  # card runs at 100 Hz


def drive_car_state(car_model: CAR, lka_states: list[int], warm_state: int = LKA_STATE_ACTIVE
                    ) -> tuple[list[structs.CarState], ToyotaCarState]:
  """Feed real EPS_STATUS bytes to the production CAN parsers so LKA_STATE is decoded the same way
  it is on the vehicle, and return the published CarState for every scenario frame. The parser
  needs a few messages before its decoded values are usable, so the scenario is preceded by a
  warm-up at warm_state whose frames are not returned."""
  CP = ToyotaCarInterface.get_non_essential_params(car_model)
  cs = ToyotaCarState(CP, structs.CarParamsSP())
  parsers = cs.get_can_parsers(CP, structs.CarParamsSP())
  packer = CANPacker(DBC[car_model][Bus.pt])

  def pack(name: str, values: dict) -> tuple[int, bytes, int]:
    return packer.make_can_msg(name, 0, values)

  outs = []
  for i, lka_state in enumerate([warm_state] * 10 + lka_states):
    frames = [pack("EPS_STATUS", {"LKA_STATE": lka_state}),
              pack("STEER_TORQUE_SENSOR", {"STEER_ANGLE": 0, "STEER_TORQUE_DRIVER": 0, "STEER_TORQUE_EPS": 0,
                                           "STEER_ANGLE_INITIALIZING": 0}),
              pack("GEAR_PACKET", {"GEAR": 2}),
              pack("BRAKE_MODULE", {"BRAKE_PRESSED": 0}),
              pack("BODY_CONTROL_STATE", {"DOOR_OPEN_FL": 0, "DOOR_OPEN_FR": 0, "DOOR_OPEN_RL": 0,
                                          "DOOR_OPEN_RR": 0, "SEATBELT_DRIVER_UNLATCHED": 0})]
    parsers[Bus.pt].update([[i * FRAME_NANOS, frames]])
    out = cs.update(parsers)[0]
    if i >= 10:
      outs.append(out)
  return outs, cs


class ToyotaFakeCarState:
  """The controller reads the EPS arm state off the CarState instance (like pcm_acc_status and
  lkas_hud), not off the published CarState: no capnp field carries it."""

  def __init__(self, lka_state: int, v_ego: float = 12.0, driver_torque: float = 0.0):
    self.out = structs.CarState()
    self.out.vEgo = v_ego
    self.out.vEgoRaw = v_ego
    self.out.aEgo = 0.0
    self.out.steeringTorque = driver_torque
    self.out.steeringTorqueEps = 0
    self.out.steeringRateDeg = 0.0
    self.out.steeringAngleDeg = 0.0
    self.out.steeringAngleOffsetDeg = 0.0
    self.out.gearShifter = GearShifter.drive
    self.out.cruiseState.enabled = True
    self.lka_state = lka_state
    self.pcm_acc_status = 0
    self.acc_type = 0
    self.gvc = 0.0
    self.lkas_hud = {}
    self.pcm_follow_distance = 0


class SteerWire:
  """Decodes what the controller actually put on the wire, since STEER_REQUEST only exists there."""

  PRIMING_FRAMES = 2

  def __init__(self, car_model: CAR):
    self.parser = CANParser(DBC[car_model][Bus.pt], [("STEERING_LKA", 0)], 0)
    self.request = 0
    self.torque = 0
    # a CANParser only reports decoded values after its first valid reception of a message, so
    # prime it; the frames under test are sent from PRIMING_FRAMES onwards
    packer = CANPacker(DBC[car_model][Bus.pt])
    for i in range(self.PRIMING_FRAMES):
      msg = packer.make_can_msg("STEERING_LKA", 0, {"STEER_REQUEST": 1, "STEER_TORQUE_CMD": 0, "SET_ME_1": 1})
      self.parser.update([[i * FRAME_NANOS, [msg]]])

  def update(self, frame_nanos: int, can_sends: list) -> None:
    self.parser.update([[frame_nanos, [(a, d, b) for a, d, b in can_sends]]])
    vl = self.parser.vl["STEERING_LKA"]
    self.request = int(vl["STEER_REQUEST"])
    self.torque = int(vl["STEER_TORQUE_CMD"])


class TestToyotaEpsFaultDebouncing(unittest.TestCase):
  def test_active_state_is_never_a_fault(self):
    outs, _ = drive_car_state(TORQUE_CAR, [LKA_STATE_ACTIVE] * 400)
    self.assertFalse(any(o.steerFaultTemporary or o.steerFaultPermanent for o in outs))

  def test_debounced_fault_states_only_report_when_persistent(self):
    for lka_state in (LKA_STATE_FAULT_TEMP, LKA_STATE_FAULT_TEMP2, LKA_STATE_FAULT_TEMP3):
      with self.subTest(lka_state=lka_state):
        self.assertIn(lka_state, carstate.TEMP_STEER_FAULTS_DEBOUNCED)
        outs, cs = drive_car_state(TORQUE_CAR, [lka_state] * (carstate.TEMP_STEER_FAULT_DEBOUNCE_FRAMES + 50))
        # the counter counts the frame it first sees the fault, so the Nth consecutive frame faults
        first_fault = next(i for i, o in enumerate(outs) if o.steerFaultTemporary)
        self.assertEqual(first_fault, carstate.TEMP_STEER_FAULT_DEBOUNCE_FRAMES - 1)
        self.assertTrue(all(o.steerFaultTemporary for o in outs[first_fault:]))
        self.assertFalse(any(o.steerFaultPermanent for o in outs))
        self.assertEqual(cs.temp_steer_fault_counter, len(outs))

  def test_fault_state_gaps_restart_the_debounce(self):
    # the observed glitch pattern: LKA_STATE 9 runs of ~1.3 s between active frames. Never fault
    frames = [LKA_STATE_ACTIVE if i % 5 == 0 else LKA_STATE_FAULT_TEMP for i in range(600)]
    outs, cs = drive_car_state(TORQUE_CAR, frames)
    self.assertFalse(any(o.steerFaultTemporary or o.steerFaultPermanent for o in outs))
    self.assertLess(cs.temp_steer_fault_counter, carstate.TEMP_STEER_FAULT_DEBOUNCE_FRAMES)

  def test_states_the_edge_cannot_fix_are_not_debounced_into_the_temp_fault(self):
    # 0 and 11 keep the existing immediate behaviour; standby (1) stays fault-free on purpose so a
    # lost EPS does not announce itself - that case is what the re-arm logic is for
    for lka_state in (0, 11):
      with self.subTest(lka_state=lka_state):
        outs, _ = drive_car_state(TORQUE_CAR, [LKA_STATE_ACTIVE] * 5 + [lka_state] * 5)
        self.assertTrue(outs[5].steerFaultTemporary)
    for lka_state in (LKA_STATE_STANDBY, LKA_STATE_UNAVAILABLE):
      with self.subTest(lka_state=lka_state):
        outs, _ = drive_car_state(TORQUE_CAR, [lka_state] * (carstate.TEMP_STEER_FAULT_DEBOUNCE_FRAMES + 50))
        self.assertFalse(any(o.steerFaultTemporary for o in outs))

  def test_permanent_fault_debounce_unchanged(self):
    outs, _ = drive_car_state(TORQUE_CAR, [LKA_STATE_UNAVAILABLE] * (carstate.PERM_STEER_FAULT_DEBOUNCE_FRAMES + 10))
    first_fault = next(i for i, o in enumerate(outs) if o.steerFaultPermanent)
    self.assertEqual(first_fault, carstate.PERM_STEER_FAULT_DEBOUNCE_FRAMES - 1)
    self.assertFalse(any(o.steerFaultTemporary for o in outs))

  def test_reporting_the_fault_waits_for_every_rearm_attempt(self):
    # controlsd gates CC.latActive on steerFaultTemporary and the re-arm logic needs latActive, so a
    # debounced fault must not withdraw lateral control before every scheduled window has finished
    CP = ToyotaCarInterface.get_non_essential_params(TORQUE_CAR)
    dbc = DBC[TORQUE_CAR][Bus.pt]
    cc_obj = ToyotaCarController({Bus.pt: dbc, Bus.cam: dbc}, CP, structs.CarParamsSP())
    last_window_end = cc_obj.EPS_REARM_FRAMES * cc_obj.EPS_REARM_MAX_ATTEMPTS + cc_obj.EPS_REARM_RESET_FRAMES
    self.assertGreater(carstate.TEMP_STEER_FAULT_DEBOUNCE_FRAMES, last_window_end)


class TestToyotaEpsRearm(unittest.TestCase):
  def _make(self, car_model: CAR = TORQUE_CAR) -> tuple[ToyotaCarController, SteerWire]:
    CP = ToyotaCarInterface.get_non_essential_params(car_model)
    dbc = DBC[car_model][Bus.pt]
    return ToyotaCarController({Bus.pt: dbc, Bus.cam: dbc}, CP, structs.CarParamsSP()), SteerWire(car_model)

  @staticmethod
  def _car_control(torque: float, lat_active: bool):
    cc = structs.CarControl()
    cc.enabled = False  # MADS-only lateral: openpilot's own enabled flag is not what demands torque
    cc.latActive = lat_active
    cc.actuators.torque = torque
    cc.actuators.longControlState = LongControlState.off
    return cc.as_reader()

  def _run(self, cc_obj, wire, states: list[int], torque: float = 0.5, lat_active: bool = True) -> list[dict]:
    log = []
    for i, lka_state in enumerate(states):
      frame_nanos = (i + SteerWire.PRIMING_FRAMES) * FRAME_NANOS
      _, can_sends = cc_obj.update(self._car_control(torque, lat_active), structs.CarControlSP(),
                                   ToyotaFakeCarState(lka_state), frame_nanos)
      wire.update(frame_nanos, can_sends)
      log.append({"req": wire.request, "cmd": wire.torque, "attempts": cc_obj.eps_rearm_attempts})
    return log

  def test_stalled_demand_gets_one_request_edge(self):
    cc_obj, wire = self._make()
    frames = cc_obj.EPS_REARM_FRAMES
    log = self._run(cc_obj, wire, [LKA_STATE_STANDBY] * (frames + cc_obj.EPS_REARM_RESET_FRAMES + 20))

    # nothing is disturbed before the stall is long enough to be worth interrupting
    self.assertTrue(all(f["req"] == 1 for f in log[:frames - 1]))
    self.assertEqual(cc_obj.eps_rearm_attempts, 1)

    # ... and then a real falling edge, long enough for the EPS's 25 Hz status to see it. The
    # EPS_REARM_FRAMES-th consecutive frame triggers it, and torque+request stay down for the
    # whole window before the next frame presents the rising edge
    zeroed = [i for i, f in enumerate(log) if f["req"] == 0]
    self.assertEqual(len(zeroed), cc_obj.EPS_REARM_RESET_FRAMES)
    self.assertEqual(min(zeroed), frames - 1)
    self.assertTrue(all(f["cmd"] == 0 for f in log[frames - 1:frames - 1 + cc_obj.EPS_REARM_RESET_FRAMES]))
    self.assertEqual(log[frames - 1 + cc_obj.EPS_REARM_RESET_FRAMES]["req"], 1)

  def test_active_eps_is_never_interrupted(self):
    cc_obj, wire = self._make()
    log = self._run(cc_obj, wire, [LKA_STATE_ACTIVE] * (cc_obj.EPS_REARM_FRAMES * 4))
    self.assertTrue(all(f["req"] == 1 and f["cmd"] != 0 for f in log))
    self.assertEqual(cc_obj.eps_rearm_attempts, 0)

  def test_no_window_without_lateral_demand(self):
    cc_obj, wire = self._make()
    self._run(cc_obj, wire, [LKA_STATE_FAULT_TEMP] * (cc_obj.EPS_REARM_FRAMES * 3), lat_active=False)
    self.assertEqual(cc_obj.eps_rearm_attempts, 0)

  def test_rearms_even_when_little_torque_is_demanded(self):
    # lateral demand alone is the condition, not torque magnitude: in the recorded 9.6 s episode the
    # commanded torque was under 50 for 543 of its 913 frames, so a magnitude gate would never have
    # acted at all
    cc_obj, wire = self._make()
    log = self._run(cc_obj, wire, [LKA_STATE_STANDBY] * (cc_obj.EPS_REARM_FRAMES + cc_obj.EPS_REARM_RESET_FRAMES + 5),
                    torque=10 / cc_obj.params.STEER_MAX)
    self.assertTrue(all(f["cmd"] != 0 for f in log[:cc_obj.EPS_REARM_FRAMES - 1]))
    self.assertEqual(sum(1 for f in log if f["req"] == 0), cc_obj.EPS_REARM_RESET_FRAMES)

  def test_no_window_for_states_an_edge_cannot_recover(self):
    for lka_state in (LKA_STATE_UNAVAILABLE, LKA_STATE_PERM_FAULT):
      with self.subTest(lka_state=lka_state):
        self.assertNotIn(lka_state, LKA_STATE_REARM)
        cc_obj, wire = self._make()
        self._run(cc_obj, wire, [lka_state] * (cc_obj.EPS_REARM_FRAMES * 3))
        self.assertEqual(cc_obj.eps_rearm_attempts, 0)

  def test_no_window_for_angle_control(self):
    # LTA uses LTA_STATE and angle commands; this torque path must stay untouched
    cc_obj, wire = self._make(ANGLE_CAR)
    self._run(cc_obj, wire, [LKA_STATE_STANDBY] * (cc_obj.EPS_REARM_FRAMES * 3))
    self.assertEqual(cc_obj.eps_rearm_attempts, 0)

  def _windows(self, log: list[dict]) -> int:
    return sum(1 for prev, cur in zip([{"req": 1}] + log[:-1], log, strict=True) if cur["req"] == 0 and prev["req"] == 1)

  def test_close_windows_then_backoff_instead_of_giving_up(self):
    cc_obj, wire = self._make()
    frames = cc_obj.EPS_REARM_FRAMES
    window = cc_obj.EPS_REARM_RESET_FRAMES
    # three attempts spaced by one detection period each, plus more time that the backoff must cover
    log = self._run(cc_obj, wire, [LKA_STATE_FAULT_TEMP] * (frames * 4 + window * 3 + 200))
    self.assertEqual(self._windows(log), cc_obj.EPS_REARM_MAX_ATTEMPTS)
    self.assertGreater(cc_obj.eps_rearm_cooldown, 0)

    # backoff expires and one more window is offered; giving up would leave an EPS that recovers
    # late unsteered for the rest of the trip
    log = self._run(cc_obj, wire, [LKA_STATE_FAULT_TEMP] * (cc_obj.EPS_REARM_BACKOFF_FRAMES + window + 5))
    self.assertEqual(self._windows(log), 1)

  def test_recovered_eps_clears_the_backoff(self):
    cc_obj, wire = self._make()
    frames = cc_obj.EPS_REARM_FRAMES
    self._run(cc_obj, wire, [LKA_STATE_FAULT_TEMP] * (frames * 3))
    self.assertGreater(cc_obj.eps_rearm_attempts + cc_obj.eps_rearm_cooldown, 0)

    # a recovered EPS clears both counters; let the in-flight window finish first so the ACTIVE
    # frames below are all real requests
    self._run(cc_obj, wire, [LKA_STATE_ACTIVE] * (cc_obj.EPS_REARM_RESET_FRAMES + 5))
    self.assertEqual((cc_obj.eps_rearm_attempts, cc_obj.eps_rearm_cooldown), (0, 0))

    window = cc_obj.EPS_REARM_RESET_FRAMES
    log = self._run(cc_obj, wire, [LKA_STATE_STANDBY] * (frames + window + 5))
    self.assertEqual(self._windows(log), 1)


if __name__ == "__main__":
  unittest.main()
