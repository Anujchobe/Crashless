from scripts.carla_report import sign_test


def test_sign_test_three_of_three():
    w, n, p = sign_test({"0": 0.83, "1": 0.55, "2": 0.87}, {"0": 0.68, "1": 0.01, "2": 0.65})
    assert (w, n) == (3, 3) and p == 0.125


def test_sign_test_drops_ties_and_unpaired_seeds():
    w, n, p = sign_test({"0": 0.5, "1": 0.9, "3": 0.9}, {"0": 0.5, "1": 0.2, "2": 0.1})
    assert (w, n) == (1, 1) and p == 0.5


def test_sign_test_five_of_five():
    a = {str(i): 1.0 for i in range(5)}
    b = {str(i): 0.0 for i in range(5)}
    assert sign_test(a, b)[2] == 1 / 32
