"""USB discovery must reject incomplete IDs rather than invent a pairing."""
import pytest

from translitter.audio import pair_devices


@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize("malformed", ["usb-headset_A", "usb-headset_A.", "usb-.analog-stereo"])
def test_incomplete_endpoint_id_is_not_selectable(direction, malformed):
    identity = "usb-" if malformed.startswith("usb-.") else "usb-headset_A"
    inputs = [(f"alsa_input.{identity}.analog-stereo", "A")]
    outputs = [(f"alsa_output.{identity}.analog-stereo", "A")]
    if direction == "input":
        inputs = [(f"alsa_input.{malformed}", "A")]
    else:
        outputs = [(f"alsa_output.{malformed}", "A")]
    # A separate well-formed device must remain available after rejection.
    inputs.append(("alsa_input.usb-headset_B.analog-stereo", "B"))
    outputs.append(("alsa_output.usb-headset_B.analog-stereo", "B"))
    assert [pair.key for pair in pair_devices(inputs, outputs)] == ["usb-headset_B"]


@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_ambiguous_profiles_are_excluded_in_either_direction(direction, duplicate, reverse):
    inputs = [("alsa_input.usb-headset_A.analog-stereo", "A"),
              ("alsa_input.usb-headset_B.analog-stereo", "B")]
    outputs = [("alsa_output.usb-headset_A.analog-stereo", "A"),
               ("alsa_output.usb-headset_B.analog-stereo", "B")]
    endpoints = inputs if direction == "input" else outputs
    endpoints.append(endpoints[0] if duplicate else (f"alsa_{direction}.usb-headset_A.mono-fallback", "A"))
    if reverse:
        inputs.reverse()
        outputs.reverse()
    assert [pair.key for pair in pair_devices(inputs, outputs)] == ["usb-headset_B"]


@pytest.mark.parametrize("identity", ["pci-internal", "bluez-headset", "virtual-usb-headset", "", "usb"])
def test_non_usb_endpoints_are_not_offered(identity):
    assert pair_devices([(f"alsa_input.{identity}.analog-stereo", "USB Headset")],
                        [(f"alsa_output.{identity}.analog-stereo", "USB Headset")]) == []


def test_distinct_dotted_device_identities_and_different_profiles_are_preserved():
    identities = ["usb-vendor.model_serial_B-00", "usb-vendor.model_serial_A-00"]
    inputs = [(f"alsa_input.{identity}.mono-fallback", "Identical USB Headset") for identity in identities]
    outputs = [(f"alsa_output.{identity}.analog-stereo", "Identical USB Headset") for identity in reversed(identities)]
    pairs = pair_devices(inputs, outputs)
    assert [pair.key for pair in pairs] == sorted(identities)
    for pair in pairs:
        assert pair.microphone_id == f"alsa_input.{pair.key}.mono-fallback"
        assert pair.output_id == f"alsa_output.{pair.key}.analog-stereo"
        assert pair.label.endswith(pair.key.rsplit("_", 1)[1].removesuffix("-00")[-6:])
    assert pair_devices(inputs, outputs[:1])[0].key == identities[1]


@pytest.mark.parametrize("wrong_side", ["input", "output"])
def test_monitor_or_wrong_direction_ids_never_replace_missing_endpoint(wrong_side):
    inputs = [("alsa_input.usb-headset_A.analog-stereo", "A")]
    outputs = [("alsa_output.usb-headset_A.analog-stereo", "A")]
    if wrong_side == "input":
        inputs = [("alsa_output.usb-headset_A.monitor", "A monitor")]
    else:
        outputs = [("alsa_input.usb-headset_A.analog-stereo", "A microphone")]
    assert pair_devices(inputs, outputs) == []
