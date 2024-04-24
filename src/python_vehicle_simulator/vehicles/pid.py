class PID:
    """Calculates and tracks a PID controller.

    Attributes:
        Kp: The proportional part gain.
        Ki: The integral part gain.
        Kd: The derivative part gain.
    """

    def __init__(self, Kp=1, Ki=1, Kd=1, clamp=160):
        """Setup PID gains and class variables."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.clamp = clamp
        self.integral = 0
        self.prev_error = None
        self.value = 0

    def update(self, error, dt):
        """Calculate the PID controller result.

        Args:
            error: The error value between the setpoint and the current point.
            dt: The time passed between the last iteration and the current.
        """
        def clamp_val(val, lim):
            if val > lim:
                val = lim
            elif val < lim*-1:
                val = lim * -1
            return val
        # print error, dt
        # print "Integral calc:", self.integral, "+", error, "*", dt
        self.integral = self.integral + error * dt
        if dt == 0 or self.prev_error == None:
            derivative = 0
        else:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error
        # print "Error:", error
        p = error * self.Kp
        i = self.integral * self.Ki
        self.integral = clamp_val(self.integral, self.clamp)
        d = derivative * self.Kd
        self.value = p + i + d
        self.value = clamp_val(self.value, self.clamp)
        # print("P:", p, "I:", i, "D:", d, "Val:", self.value, "Err:", error)
        return self.value

    def get_val(self):
        return self.value

    def reset(self):
        self.integral = 0
        self.prev_error = None
        self.value = 0
