import unittest
import numpy as np
from numpy.testing import assert_allclose
import pickle as pk
import wgan 


seed = 10707
TOLERANCE = 1e-5

# to run one test: python -m unittest tests.
# to run all tests: python -m unittest tests


with open('tests.pk', "rb") as f:
    tests = pk.load(f)


class test_discriminator_forward(unittest.TestCase):
    def test(self):
        np.random.seed(seed)
        discriminator = wgan.Discriminator()
        x = np.random.rand(64, 28, 28, 1)
        assert_allclose(discriminator.forward(x), tests[0], atol=TOLERANCE)

class test_discriminator_backward(unittest.TestCase):
    def test(self):
        np.random.seed(seed)
        discriminator = wgan.Discriminator()
        x = np.random.rand(64, 28, 28, 1)
        logit = discriminator.forward(x)
        # logit = np.random.rand(64, 1)
        discriminator.backward(logit, x, -1)

        assert_allclose(discriminator.W0, tests[1][0], atol=TOLERANCE)
        assert_allclose(discriminator.b0, tests[1][1], atol=TOLERANCE)
        assert_allclose(discriminator.W1, tests[1][2], atol=TOLERANCE)
        assert_allclose(discriminator.b1, tests[1][3], atol=TOLERANCE)
        assert_allclose(discriminator.W2, tests[1][4], atol=TOLERANCE)
        assert_allclose(discriminator.b2, tests[1][5], atol=TOLERANCE)

class test_discriminator_weight_clipping(unittest.TestCase):
    def test(self):
        np.random.seed(seed)
        discriminator = wgan.Discriminator()
        discriminator.weight_clipping()

        assert_allclose(discriminator.W0, tests[2][0], atol=TOLERANCE)
        assert_allclose(discriminator.b0, tests[2][1], atol=TOLERANCE)
        assert_allclose(discriminator.W1, tests[2][2], atol=TOLERANCE)
        assert_allclose(discriminator.b1, tests[2][3], atol=TOLERANCE)
        assert_allclose(discriminator.W2, tests[2][4], atol=TOLERANCE)
        assert_allclose(discriminator.b2, tests[2][5], atol=TOLERANCE)

class test_generator_forward(unittest.TestCase):
    def test(self):
        np.random.seed(seed)
        generator = wgan.Generator()
        z = np.random.rand(64, 100)
        out = generator.forward(z)

        assert_allclose(out, tests[3][0], atol=TOLERANCE)

class test_generator_backward(unittest.TestCase):
    def test(self):
        np.random.seed(seed)
        generator = wgan.Generator()
        discriminator = wgan.Discriminator()

        z = np.random.rand(64, 100)
        fake_input = generator.forward(z)
        fake_logit = discriminator.forward(fake_input)

        generator.backward(fake_logit, fake_input, discriminator)

        assert_allclose(generator.W0, tests[4][0], atol=TOLERANCE)
        assert_allclose(generator.b0, tests[4][1], atol=TOLERANCE)
        assert_allclose(generator.W1, tests[4][2], atol=TOLERANCE)
        assert_allclose(generator.b1, tests[4][3], atol=TOLERANCE)
        assert_allclose(generator.W2, tests[4][4], atol=TOLERANCE)
        assert_allclose(generator.b2, tests[4][5], atol=TOLERANCE)
        assert_allclose(generator.W3, tests[4][6], atol=TOLERANCE)
        assert_allclose(generator.b3, tests[4][7], atol=TOLERANCE)

