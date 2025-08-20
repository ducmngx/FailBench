# MuJoCo Failure Injection Methods - Detailed Explanation

## 1. External Disturbance Forces (`data.qfrc_applied`)

### How it Works
MuJoCo's physics engine computes joint torques through the equation:
```
τ_total = τ_control + τ_applied + τ_passive + τ_constraint
```

Where:
- `τ_control`: Your position controller output
- `τ_applied`: External forces you inject via `data.qfrc_applied`
- `τ_passive`: Springs, dampers, friction
- `τ_constraint`: Contact forces, joint limits

### Implementation Details
```python
def inject_disturbance(model, data, joint_name, disturbance_torque):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    # This adds directly to the joint's generalized force
    data.qfrc_applied[joint_id] += disturbance_torque
```

### Physics Behind It
- `data.qfrc_applied` is added **after** your controller computes its output
- It directly affects the joint's equation of motion: `M(q)q̈ + C(q,q̇) = τ_total`
- The disturbance appears as an external torque that your controller must compensate for
- This simulates real-world disturbances like external forces, wind, collisions

### When to Use
- Simulating external disturbances (robot being pushed)
- Testing controller robustness to unexpected torques
- Modeling environmental forces (wind, vibrations)

---

## 2. Actuator Parameter Modification

### 2a. Position Gain Reduction (`model.actuator_gainprm`)

#### How it Works
MuJoCo position actuators compute torque as:
```
τ_actuator = kp * (θ_target - θ_current) + kd * (ω_target - ω_current)
```

Where `kp` and `kd` are stored in `model.actuator_gainprm[actuator_id, 0]` and `model.actuator_gainprm[actuator_id, 1]`

#### Implementation
```python
# Reduce actuator strength by 50%
original_kp = model.actuator_gainprm[actuator_id, 0]
model.actuator_gainprm[actuator_id, 0] = original_kp * 0.5
```

#### Physics Behind It
- Lower `kp` means weaker position control
- The actuator produces less torque for the same position error
- System becomes more compliant and slower to respond
- Simulates motor degradation, power loss, or mechanical wear

### 2b. Actuator Bias (`data.actuator_bias`)

#### How it Works
The actuator equation becomes:
```
τ_actuator = kp * (θ_target + bias - θ_current) + kd * (ω_target - ω_current)
```

#### Implementation
```python
# Simulate stuck actuator at current position
current_pos = data.qpos[joint_id]
data.actuator_bias[actuator_id] = stuck_offset
```

#### Physics Behind It
- Bias shifts the "zero error" point of the controller
- Positive bias makes the actuator think it needs to move further
- Simulates sensor calibration errors, mechanical backlash, or stuck components

### 2c. Control Signal Noise

#### How it Works
```python
# Add noise directly to control signal
data.ctrl[actuator_id] += np.random.normal(0, noise_std)
```

#### Physics Behind It
- Noise is added to the target position before the actuator computes torque
- Higher frequency noise excites system dynamics
- Simulates electrical interference, quantization errors, communication dropouts

---

## 3. Joint Property Modifications

### 3a. Joint Friction (`model.dof_frictionloss`)

#### How it Works
MuJoCo adds viscous damping to joint dynamics:
```
τ_friction = -frictionloss * q̇
```

#### Implementation
```python
# Double the joint friction
model.dof_frictionloss[joint_id] *= 2.0
```

#### Physics Behind It
- Higher friction requires more torque to maintain motion
- Affects both forward and backward motion equally
- Simulates bearing wear, lubricant loss, or contamination

### 3b. Joint Stiffness (`model.dof_stiffness`)

#### How it Works
Adds spring force proportional to joint displacement from reference:
```
τ_spring = -stiffness * (q - q_ref)
```

#### Implementation
```python
# Add joint stiffness (makes joint resist motion)
model.dof_stiffness[joint_id] = high_stiffness_value
```

#### Physics Behind It
- Joint tries to return to reference position (usually 0)
- Higher stiffness makes joint harder to move
- Simulates joint damage, contamination, or mechanical binding

### 3c. Joint Range Limits (`model.jnt_range`)

#### How it Works
MuJoCo enforces hard constraints when joint exceeds limits:
```python
if q < q_min or q > q_max:
    # Apply constraint forces to keep joint in bounds
    apply_constraint_force()
```

#### Implementation
```python
# Restrict joint to small range around current position
current_pos = data.qpos[joint_id]
model.jnt_range[joint_id, 0] = current_pos - 0.1  # Lower limit
model.jnt_range[joint_id, 1] = current_pos + 0.1  # Upper limit
```

#### Physics Behind It
- Constraint solver applies forces to prevent limit violation
- Can cause sudden stops and force spikes
- Simulates mechanical stops, cable length limits, or safety constraints

---

## 4. Sensor Feedback Corruption

### How it Works
Your controller uses sensor readings to compute errors:
```python
position_error = target_position - measured_position
velocity_error = target_velocity - measured_velocity
control_output = kp * position_error + kd * velocity_error
```

If `measured_position` is corrupted, the controller makes wrong decisions.

### 4a. Sensor Noise
```python
def add_sensor_noise(true_value, noise_std):
    return true_value + np.random.normal(0, noise_std)

corrupted_pos = add_sensor_noise(data.qpos[joint_id], 0.01)  # 0.01 rad noise
```

**Physics**: Controller responds to noise as if it were real position changes, causing unnecessary corrections.

### 4b. Sensor Bias
```python
def add_sensor_bias(true_value, bias):
    return true_value + bias

corrupted_pos = add_sensor_bias(data.qpos[joint_id], 0.1)  # 0.1 rad bias
```

**Physics**: Controller consistently thinks joint is 0.1 rad away from true position, causing steady-state error.

### 4c. Sensor Dropout
```python
class SensorDropout:
    def __init__(self):
        self.last_valid_reading = 0
        self.dropout_active = False
    
    def get_reading(self, true_value, dropout_probability=0.01):
        if np.random.random() < dropout_probability:
            self.dropout_active = True
            return self.last_valid_reading  # Return stale data
        else:
            self.dropout_active = False
            self.last_valid_reading = true_value
            return true_value
```

**Physics**: Controller uses outdated information, leading to incorrect control actions based on old state.

### 4d. Sensor Scaling Error
```python
def add_scaling_error(true_value, scale_factor):
    return true_value * scale_factor

corrupted_pos = add_scaling_error(data.qpos[joint_id], 0.9)  # 10% scaling error
```

**Physics**: Controller thinks joint moved less/more than actual, causing over/under-correction.

---

## 5. External Force Application (`data.xfrc_applied`)

### How it Works
Forces applied to bodies propagate through kinematic chain:
```
F_body → τ_joint = J^T * F_body
```

Where `J` is the Jacobian matrix mapping joint velocities to body velocities.

### Implementation
```python
def inject_external_force(model, data, body_name, force, torque=None):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    data.xfrc_applied[body_id, :3] += force      # Linear force
    if torque is not None:
        data.xfrc_applied[body_id, 3:] += torque  # Angular torque
```

### Physics Behind It
- Forces applied to end-effector create torques at all upstream joints
- Closer joints experience higher torques (longer moment arms)
- Direction of applied force determines which joints are most affected
- Simulates contact forces, tool interactions, payload changes

---

## 6. Time-Based Failure Patterns

### Gradual Degradation
```python
def gradual_degradation(model, data, degradation_rate=0.1):
    t = data.time
    degradation_factor = max(0.1, 1.0 - degradation_rate * t / 60.0)
    model.actuator_gainprm[:, 0] *= degradation_factor
```

**Physics**: Simulates wear over time by continuously reducing actuator gains.

### Intermittent Failures
```python
def intermittent_failure(data, failure_probability=0.01):
    if np.random.random() < failure_probability:
        data.ctrl[:] = 0  # Complete actuator shutdown
```

**Physics**: Randomly removes all control authority, forcing system to rely on passive dynamics.

### Environmental Disturbances
```python
def environmental_disturbance(model, data, body_name):
    t = data.time
    # Sinusoidal wind force
    wind_force = 5 * np.sin(0.5 * t) * np.array([1, 0, 0])
    inject_external_force(model, data, body_name, wind_force)
```

**Physics**: Applies time-varying external forces that challenge controller's disturbance rejection.

---

## Integration with Your Position Controller

### Complete Example
```python
class RobustnessTester:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.original_gains = model.actuator_gainprm.copy()
        self.sensor_noise_std = 0.0
        self.disturbance_torques = np.zeros(model.nv)
    
    def step(self, target_positions):
        # 1. Apply sensor corruption
        measured_positions = self.corrupt_sensors(self.data.qpos)
        
        # 2. Compute your normal position control
        position_errors = target_positions - measured_positions
        control_torques = self.compute_pd_control(position_errors)
        
        # 3. Apply actuator failures
        self.apply_actuator_degradation()
        
        # 4. Set control signals (potentially corrupted)
        self.data.ctrl[:] = control_torques + self.get_control_noise()
        
        # 5. Apply external disturbances
        self.data.qfrc_applied[:] = self.disturbance_torques
        
        # 6. Step physics (MuJoCo handles the rest)
        mujoco.mj_step(self.model, self.data)
    
    def corrupt_sensors(self, true_positions):
        return true_positions + np.random.normal(0, self.sensor_noise_std, len(true_positions))
    
    def compute_pd_control(self, position_errors):
        kp, kd = 100, 10
        velocity_errors = -self.data.qvel  # Assuming zero target velocity
        return kp * position_errors + kd * velocity_errors
    
    def apply_actuator_degradation(self):
        # Gradually reduce actuator strength
        degradation = 0.999  # 0.1% degradation per step
        self.model.actuator_gainprm[:, 0] *= degradation
```

This framework lets you systematically test how your position control planner responds to various failure modes while maintaining the underlying physics fidelity of MuJoCo.