//! A proportional-integral-derivative (PID) controller.
#![no_std]

use core::cmp::PartialOrd;
use core::ops::{Add, Mul, Neg, Sub};
use num_traits::Zero;
use num_traits::One;
use num_traits::Signed;
use num_traits::AsPrimitive;

pub struct Pid<T>
where
    T: PartialOrd + Copy + Clone,
{
    /// Proportional gain.
    kp: T,
    /// Integral gain.
    ki: T,
    /// Derivative gain.
    kd: T,
    /// Feed forward gain.
    kff: T,
    /// Limit of output
    limit: Limit<T>,

    setpoint: T,
    prev_measurement: T,
    /// `integral_term = sum[error(t) * ki(t)] (for all t)`
    integral_term: T,
    /// internal error signum
    err_sat_sig: T,
}

pub struct ControlOutput<T : Signed> {
    /// Contribution of the P term to the output.
    pub p: T,
    /// Contribution of the I term to the output.
    /// `i = sum[error(t) * ki(t)] (for all t)`
    pub i: T,
    /// Contribution of the D term to the output.
    pub d: T,
    /// Contribution of the feed forward term to the output.
    pub ff: T,
    /// Output of the PID controller.
    pub output: T,
}

impl<T> Pid<T>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Neg<Output = T>
        + Mul<Output = T>
        + Zero
        + One
        + Signed
        + AsPrimitive<T>
        + Default
        + PartialOrd
        + Copy
        + Clone,
{
    pub fn new(kp: T, ki: T, kd: T, kff: T, lower_limit: T, upper_limit: T, setpoint: T, initial_measurement: T) -> Self {
        Self {
            kp,
            ki,
            kd,
            kff,
            limit: Limit::<T>{min:lower_limit, max:upper_limit},
            setpoint,
            prev_measurement: initial_measurement,
            integral_term: T::zero(),
            err_sat_sig: T::zero(),
        }
    }

    pub fn set_limits(
        &mut self,
        limit: Limit<T>,
    ) {
        self.limit=limit;
    }

    pub fn update_pid_terms(&mut self, kp: T, ki: T, kd: T, kff: T) {
        self.kp = kp;
        self.ki = ki;
        self.kd = kd;
        self.kff = kff;
    }

    pub fn update_setpoint(&mut self, setpoint: T) {
        self.setpoint = setpoint;
    }

    /// Resets the integral term back to zero. This may drastically change the
    /// control output.
    pub fn reset_integral_term(&mut self) {
        self.integral_term = T::zero();
    }

    /// Given a new measurement, calculates the next control output.
    ///
    /// # Panics
    /// If a setpoint has not been set via `update_setpoint()`.
    pub fn next_control_output(&mut self, measurement: T) -> ControlOutput<T> {
        let error = self.setpoint - measurement;

        let p = error * self.kp;

        // Mitigate output jumps when ki(t) != ki(t-1).
        // While it's standard to use an error_integral that's a running sum of
        // just the error (no ki), because we support ki changing dynamically,
        // we store the entire term so that we don't need to remember previous
        // ki values.
        if self.err_sat_sig * error <= T::zero()  { // (self.err_sat_sig < 0 && error < 0) || (self.err_sat_sig > 0 && error > 0)
            // read https://www.embeddedrelated.com/showarticle/121.php on the 5 good reasons why we scale first and integrate later
            let tmp_integral_term = self.integral_term + error * self.ki;
            // Mitigate integral windup: Don't want to keep building up error
            // beyond what output_limit will allow.
            (self.integral_term, self.err_sat_sig) = self.limit.apply_sat(tmp_integral_term);
        }

        // Mitigate derivative kick: Use the derivative of the measurement
        // rather than the derivative of the error.
        let d = -(measurement - self.prev_measurement) * self.kd;
        self.prev_measurement = measurement;

        let ff = self.setpoint * self.kff;

        let output_unbounded = p + self.integral_term + d + ff;
        let output = self.limit.apply(output_unbounded);

        ControlOutput {
            p,
            i: self.integral_term,
            d,
            ff,
            output,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct Limit<T>
where
    T: PartialOrd + Copy + Clone,
{
    min: T,
    max: T,
}

impl<T> Limit<T>
where
    T: PartialOrd + Copy + Clone + One + Zero + Signed + AsPrimitive<T> + Sub<Output = T> + Neg<Output = T> + Mul<Output = T>
{
    pub fn new(min: T, max: T) -> Self {
        Self { min, max }
    }

    pub fn apply(&self, value: T) -> T {
        if value > self.max {
            self.max
        } else if value < self.min {
            self.min
        } else {
            value
        }
    }

    pub fn apply_sat(&self, value: T) -> (T,T) {
        if value > self.max {
            ( self.max , T::one() )
        } else if value < self.min {
            ( self.min, T::neg(T::one()))
        } else {
            ( value, T::zero())
        }
    }

}

#[cfg(test)]
mod tests {
    use crate::Limit;

    use super::Pid;

    #[test]
    fn proportional() { // kp: T, ki: T, kd: T, kff: T, lower_limit: T, upper_limit: T, setpoint: T, initial_measurement: T)
        let mut pid = Pid::new(2, 0, 0, 0, -20, 20, 10, 0);
        assert_eq!(pid.setpoint, 10);

        // Test simple proportional
        assert_eq!(pid.next_control_output(0).output, 20);

        // Test proportional limit
        pid.set_limits(Limit::new(-10, 10));
        assert_eq!(pid.next_control_output(0).output, 10);
    }

    #[test]
    fn derivative() {
        let mut pid = Pid::new(0, 0, 2, 10, -10, 10, 0, 0);

        // Test that there's no derivative since it's the first measurement
        assert_eq!(pid.next_control_output(0).output, 0);

        // Test that there's now a derivative
        assert_eq!(pid.next_control_output(5).output, -10);

        // Test derivative limit
        pid.set_limits(Limit::new(-5, 5));
        assert_eq!(pid.next_control_output(10).output, -5);
    }

    #[test]
    fn integral() {
        let mut pid = Pid::new(0, 2, 0, 0, -50, 50, 10, 0);

        // Test basic integration
        assert_eq!(pid.next_control_output(0).output, 20);
        assert_eq!(pid.next_control_output(0).output, 40);
        assert_eq!(pid.next_control_output(5).output, 50);

        // Test limit
        pid.set_limits(Limit::new(-50, 50));
        assert_eq!(pid.next_control_output(5).output, 50);
        // Test that limit doesn't impede reversal of error integral
        assert_eq!(pid.next_control_output(15).output, 40);

        // Test that error integral accumulates negative values
        let mut pid2 = Pid::new(0, 2, 0, 0, -50, 50, -10, 0);
        assert_eq!(pid2.next_control_output(0).output, -20);
        assert_eq!(pid2.next_control_output(0).output, -40);

        pid2.set_limits(Limit::new(-50, 50));
        assert_eq!(pid2.next_control_output(-5).output, -50);
        // Test that limit doesn't impede reversal of error integral
        assert_eq!(pid2.next_control_output(-15).output, -40);
    }

    #[test]
    fn output_limit() {
        let mut pid = Pid::new(1, 0, 0, 0, -1, 1, 10, 0);

        let out = pid.next_control_output(0);
        assert_eq!(out.p, 10); // 1 * 10
        assert_eq!(out.output, 1);

        let out = pid.next_control_output(20);
        assert_eq!(out.p, -10); // 1 * (10 - 20)
        assert_eq!(out.output, -1);
    }

    #[test]
    fn pid() {
        let mut pid = Pid::new(10, 1, 10, 0, -10000, 10000, 100, 0);

        let out = pid.next_control_output(0);
        assert_eq!(out.p, 1000); // 1 * 10
        assert_eq!(out.i, 100); // 0.1 * 10
        assert_eq!(out.d, 0); // -(1 * 0)
        assert_eq!(out.output, 1100);

        let out = pid.next_control_output(50);
        assert_eq!(out.p, 500); // 1 * 5
        assert_eq!(out.i, 150); // 0.1 * (10 + 5)
        assert_eq!(out.d, -500); // -(1 * 5)
        assert_eq!(out.output, 150);

        let out = pid.next_control_output(110);
        assert_eq!(out.p, -100); // 1 * -1
        assert_eq!(out.i, 140); // 0.1 * (10 + 5 - 1)
        assert_eq!(out.d, -600); // -(1 * 6)
        assert_eq!(out.output, -560);

        let out = pid.next_control_output(100);
        assert_eq!(out.p, 0); // 1 * 0
        assert_eq!(out.i, 140); // 0.1 * (10 + 5 - 1 + 0)
        assert_eq!(out.d, 100); // -(1 * -1)
        assert_eq!(out.output, 240);
    }
}
