from lib.models import create_2unit_net, run_test_model
from lib.network import Sequential


def test_2unit_model():
    """Create the 2 unit 2 layer network and test it"""
    model = create_2unit_net()
    assert isinstance(model, Sequential), f"model should be Sequential but is {type(model)}"
    assert (
        len(model.modules) == 3
    ), f"model should have 3 modules (2 layers + 1 activation function) but has {len(model.modules)}"
    run_test_model(model)


if __name__ == '__main__':
    test_2unit_model()
    print("Test complete.")
