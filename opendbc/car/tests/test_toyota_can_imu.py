"""Toyota's CAN IMU decode (KINEMATICS, 0x24) and its unpopulated-signal detection.

This message is the only attitude source left once the external IMU drops out, so two things have to
hold: the decode must match the CAN parser exactly, and a car whose ECU does not fill the fields in
must be handled as "no data" rather than as a stream of plausible-looking constants.

The frames below are real, taken from route 000000b6--4ed521d614--3 on a TOYOTA_CAMRY_TSS2, with the
values the parser decodes from them recorded alongside. Synthesising frames by hand-shifting DBC bits
is how a test ends up passing against the wrong bit layout, so real frames are used instead.
"""
import unittest

from opendbc.can.parser import CANParser
from opendbc.car import Bus
from opendbc.car.structs import CarState
from opendbc.car.toyota.carstate import CAN_IMU_FROZEN_FRAMES, CarState as ToyotaCarState
from opendbc.car.toyota.values import DBC
from opendbc.car.values import PLATFORMS

FP = "TOYOTA_CAMRY_TSS2"
DBC_NAME = DBC[FP][Bus.pt]
DT_NS = 10_000_000  # carState runs at 100 Hz

# (frame, decoded YAW_RATE in deg/s, decoded ACCEL_X in m/s^2). The yaw value moves across these,
# which is exactly what the unpopulated check keys on.
REAL_FRAMES = [
  (bytes.fromhex("020801f9420c80fe"), 1.880, -0.2506),
  (bytes.fromhex("020801f9420d80ff"), 1.880, -0.2506),
  (bytes.fromhex("020701f9420d80fe"), 1.636, -0.2506),
  (bytes.fromhex("020701f9420c80fd"), 1.636, -0.2506),
  (bytes.fromhex("020701fa420c80fe"), 1.636, -0.2147),
  (bytes.fromhex("020701f8420c80fc"), 1.636, -0.2864),
]


class _FakeCP:
  """Just enough of a CANParser for CarState.update_can_imu."""

  def __init__(self, parser: CANParser):
    self.vl = parser.vl
    self.dbc = parser.dbc
    self._can_valid = True

  @property
  def can_valid(self) -> bool:
    return self._can_valid


def _parser():
  return CANParser(DBC_NAME, [("KINEMATICS", float('nan'))], 0)


def _car_state_under_test():
  """A CarState carrying only the three fields update_can_imu touches."""
  cs = ToyotaCarState.__new__(ToyotaCarState)
  cs._imu_last_yaw = None
  cs._imu_yaw_frozen_frames = 0
  cs._imu_yaw_unpopulated = False
  return cs


class TestToyotaCanImuDecode(unittest.TestCase):
  def test_moving_signal_is_published(self):
    parser, cp, cs = _parser(), None, _car_state_under_test()
    cp = _FakeCP(parser)
    t = 1_000_000_000
    for frame, _, _ in REAL_FRAMES:
      parser.update([(t, [(36, frame, 0)])])
      t += DT_NS

    ret = CarState()
    cs.update_can_imu(ret, cp)

    _, last_yaw_deg, last_ax = REAL_FRAMES[-1]
    self.assertTrue(ret.imuValid, "a moving signal must be published")
    self.assertAlmostEqual(ret.yawRate, last_yaw_deg * 0.017453292519943295, places=6)
    self.assertAlmostEqual(ret.imuAccel, last_ax, places=4)

  def test_decoded_values_match_the_parser(self):
    """update_can_imu must not reinterpret the signal - it only scales and range-checks it."""
    parser, cs = _parser(), _car_state_under_test()
    cp = _FakeCP(parser)
    t = 1_000_000_000

    for frame, yaw_deg, ax in REAL_FRAMES:
      parser.update([(t, [(36, frame, 0)])])
      t += DT_NS

      direct = parser.vl["KINEMATICS"]
      self.assertAlmostEqual(direct["YAW_RATE"], yaw_deg, places=6, msg="fixture is wrong")
      self.assertAlmostEqual(direct["ACCEL_X"], ax, places=4, msg="fixture is wrong")

      ret = CarState()
      cs.update_can_imu(ret, cp)
      # the capnp fields are Float32, so compare at Float32 precision
      self.assertAlmostEqual(ret.yawRate, direct["YAW_RATE"] * 0.017453292519943295, places=8)
      self.assertAlmostEqual(ret.imuAccel, direct["ACCEL_X"], places=6)

  def test_frozen_signal_is_rejected_and_recovers_on_its_own(self):
    """An unfilled field arrives on time with a constant value; that must read as "no data"."""
    parser, cs = _parser(), _car_state_under_test()
    cp = _FakeCP(parser)

    t = 1_000_000_000
    ret = CarState()
    for _ in range(CAN_IMU_FROZEN_FRAMES + 5):
      parser.update([(t, [(36, REAL_FRAMES[0][0], 0)])])
      ret = CarState()
      cs.update_can_imu(ret, cp)
      t += DT_NS
    self.assertFalse(ret.imuValid, "a permanently frozen signal must be reported invalid")

    # the moment it moves it must come back, with no external nudge
    parser.update([(t, [(36, REAL_FRAMES[-1][0], 0)])])
    recovered = CarState()
    cs.update_can_imu(recovered, cp)
    self.assertTrue(recovered.imuValid, "the signal must recover as soon as it moves")

  def test_absurd_values_are_rejected(self):
    for yaw_deg, accel_x, why in [(400.0, 0.0, "absurd yaw rate"), (1.0, 90.0, "absurd accel")]:
      parser, cs = _parser(), _car_state_under_test()
      cp = _FakeCP(parser)
      parser.update([(1_000_000_000, [(36, REAL_FRAMES[-1][0], 0)])])
      parser.vl["KINEMATICS"]["YAW_RATE"] = yaw_deg
      parser.vl["KINEMATICS"]["ACCEL_X"] = accel_x

      ret = CarState()
      cs.update_can_imu(ret, cp)
      self.assertFalse(ret.imuValid, f"{why} must be rejected")

  def test_can_errors_yield_no_imu(self):
    parser, cs = _parser(), _car_state_under_test()
    cp = _FakeCP(parser)
    cp._can_valid = False
    parser.update([(1_000_000_000, [(36, REAL_FRAMES[0][0], 0)])])

    ret = CarState()
    cs.update_can_imu(ret, cp)
    self.assertFalse(ret.imuValid, "CAN errors must not yield IMU data")


class TestToyotaCanImuRegistration(unittest.TestCase):
  def test_every_toyota_pt_dbc_carries_the_message(self):
    """update_can_imu indexes cp.dbc.name_to_msg, so every Toyota pt DBC must define KINEMATICS.

    Checked by building the parser, which is the same lookup the code does: a DBC without the message
    raises rather than quietly yielding nothing.
    """
    missing = []
    for name, platform in PLATFORMS.items():
      if not (name.startswith("TOYOTA") or name.startswith("LEXUS")):
        continue
      try:
        CANParser(platform.config.dbc_dict[Bus.pt], [("KINEMATICS", float('nan'))], 0)
      except (RuntimeError, KeyError) as e:
        missing.append(f"{name}: {e}")
    self.assertEqual(missing, [], "these Toyota platforms cannot decode KINEMATICS")

  def test_message_is_exempt_from_can_valid(self):
    """A Toyota that never broadcasts 0x24 must still pass the parser's liveness check."""
    parser = _parser()
    self.assertIn(36, parser.message_states)
    self.assertTrue(parser.message_states[36].ignore_alive,
                    "KINEMATICS must be registered ignore-alive, or a car without it cannot drive")
    self.assertTrue(parser.message_states[36].valid(10 ** 12, bus_timeout=True))


if __name__ == "__main__":
  unittest.main()