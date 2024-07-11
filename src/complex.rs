#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64, //real part of the complex plane (a in a+bi)
    pub im: f64 //imaginary part of the complex part (bi in a+bi, where i is the imaginary unit)
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Complex {re, im}
    }
    pub fn square(self) -> Self {
        Self {
            re: (self.re * self.re) - (self.im * self.im),
            im:  2.0 * self.re * self.im,
        }
    }

    pub fn magnitude_squared(self) -> f64 {
        (self.re * self.re) + (self.im * self.im)
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;

    fn add(self, other: Complex) -> Complex {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}