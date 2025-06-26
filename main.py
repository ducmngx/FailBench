import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from controllers.diff_drive import DifferentialDriveController

xml_path = 'jackal.xml' #xml file (assumes this is in the same folder as this file)
simend = 500 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


# TODO: set this in a seperate class
linear_vel = 0
angular_vel = 0


_overlay = {}
def add_overlay(gridpos, text1, text2):

    if gridpos not in _overlay:
        _overlay[gridpos] = ["", ""]
    _overlay[gridpos][0] += text1 + "\n"
    _overlay[gridpos][1] += text2 + "\n"


def create_overlay(model, data):
    global linear_vel, angular_vel

    topleft = mj.mjtGridPos.mjGRID_TOPLEFT
    topright = mj.mjtGridPos.mjGRID_TOPLEFT
    bottomleft = mj.mjtGridPos.mjGRID_BOTTOMLEFT
    bottomright = mj.mjtGridPos.mjGRID_BOTTOMRIGHT

    add_overlay(bottomleft,"Restart",'r', )

    add_overlay(
        bottomleft,
        "Start",'s' ,
         )

    add_overlay(
        bottomleft,
        "Time",'%.2f' % data.time,
         )

    add_overlay(
        topleft,
        "Linear Vel. (up/down)",'%.2f' % linear_vel ,
         )

    add_overlay(
        topleft,
        "Angular Vel. (left/right)",'%.2f' % angular_vel,
         )


def keyboard(window, key, scancode, act, mods):
    global linear_vel, angular_vel, controller
    
    if act == glfw.PRESS and key == glfw.KEY_UP:
        linear_vel += 0.01

    if act == glfw.PRESS and key == glfw.KEY_DOWN:
        linear_vel -= 0.01

    if act == glfw.PRESS and key == glfw.KEY_LEFT:
        angular_vel -= 0.05

    if act == glfw.PRESS and key == glfw.KEY_RIGHT:
        angular_vel += 0.05
    
    controller.control(linear_vel, angular_vel)


def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname, "models", xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
cam.azimuth = 43
cam.elevation = -48 
cam.distance =  10
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

# Controller stuff
jackal_wheel_radius = 0.098
jackal_width = 0.310
left_wheel_actuators = data.ctrl[:2]
right_wheel_actuators = data.ctrl[2:]

controller = DifferentialDriveController(jackal_wheel_radius, jackal_width, left_wheel_actuators, right_wheel_actuators)


#set the controller
# mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        # controller.control(linear_vel, angular_vel)
        mj.mj_step(model, data)
        # print(data.ctrl[0],data.ctrl[1],data.ctrl[2],data.ctrl[3])


    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #create overlay
    create_overlay(model, data)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # overlay items
    for gridpos, [t1, t2] in _overlay.items():
        mj.mjr_overlay(
            mj.mjtFontScale.mjFONTSCALE_150,
            gridpos,
            viewport,
            t1,
            t2,
            context)

    # clear overlay
    _overlay.clear()

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
