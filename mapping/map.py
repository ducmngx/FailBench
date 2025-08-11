import numpy as np
# from scipy.ndimage import binary_closing

# for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

from scipy.ndimage import binary_dilation

class Map:
    """
    Map: 2d boolean grid representing whether the cell is occupied by an obstacle or not.

    Underlying map: (height x width) @ init_resolution.
    User map: same occupancy map at a lower resolution for increased speed during planning.

    say real_map (rm) is (h,w) at resolution = 100
    user_map (um) is (h, w) * (current_resolution/init_resolution)
    so current_resolution = 75
    then h,w* (75/100) = (h*0.75, w*0.75)

    okay so for 
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

    user_x      map_x           ~ occupancy calculation     -> range of cells
    (0)         0->1.33         ~ 1*[0] + 0.33*[1]          -> [floor(0), ceil(1.33))
    (1)         1.33->2.67      ~ 0.67*[1] + 0.67*[2]       -> [floor(1.33), ceil(2.67))
    (2)         2.67-> 4        ~ 0.33*[2] + 1*[3]          -> [floor(2.67), ceil(4))
    
    now if we have a 2d map these 

    (1*1, 0.33*1, 1*0.33, 0.33*0.33) -> weights

    so given x we calculate range by: x_ in range (floor(x*(init/current)), ceil((x+1)*(init/current)) )
    so given y we calculate range by: y_ in range (floor(y*(init/current)), ceil((y+1)*(init/current)) )
    
    boundaries: given x, boundaries are [x*(init/curr), (x+1)*(init/curr))

    x_ in range (floor(x*(init/current)), ceil((x+1)*(init/current)) ):
        y_ in range (floor(y*(init/current)), ceil((y+1)*(init/current)) ):

    then if (x_+1) - lower < 1 || upper - x_ < 1:  
        weigh it by diff
    then get weighted average. if weighted average > 0.5 -> occupied.
    """

    # initialization methods
    def __init__(self, type="grid", init_resolution=100):
        self.type = type
        self.init_resolution = self.current_resolution = init_resolution

        # things that should be set later
        self.raw_map = None
        self.inflated_map = None
        self.world_mins = None
        self.height = None
        self.width = None

    def set_size(self, height, width):
        assert self.type == "grid"
        self.raw_map = np.zeros((height, width), dtype=bool)
        self.height = height
        self.width = width

    def set_world_mins(self, xmin, ymin):
        # set it as [[y,x]] for ease of translation in convert_to_world_coordinates
        # remember, self.raw_map = (H, W)
        self.world_mins = np.array([[ymin, xmin]]) 

    # Map filling methods
    def set_rect_obstacles(self, x0, y0, x1, y1):
        """
        Set a rectangular obstacle on the map at the init resolution.
        """
        assert self.current_resolution == self.init_resolution
        self.raw_map[y0:y1, x0:x1] = True
   
    # data methods
    def is_cell_occupied(self, x, y):
        """
        Central method which returns whether the cell at the user resolution is occupied or not.
        If an inflated map was constructed, the inflated map will be used.
        """
        if x < 0 or y < 0 or x >= self.width*(self.current_resolution/self.init_resolution) or y >= self.height*(self.current_resolution/self.init_resolution):
            raise IndexError

        map_to_extract_from = self.raw_map if self.inflated_map is None else self.inflated_map
        if self.current_resolution == self.init_resolution:
            return map_to_extract_from[y, x]
        

        downscale_ratio = self.init_resolution/self.current_resolution
        boundary_x = (x*downscale_ratio, (x+1)*downscale_ratio)
        boundary_y = (y*downscale_ratio, (y+1)*downscale_ratio)
        weighted_sum = 0
        sum_of_weights = 0
        for x_ in range(int(x*downscale_ratio), int(math.ceil((x+1)*downscale_ratio))):
            weight = 1
            low_diff_x = x_+1 - boundary_x[0]
            upper_diff_x = boundary_x[1] - x_ 
            if low_diff_x < 1 :
                weight *= low_diff_x
            elif upper_diff_x<1:
                weight *= upper_diff_x

            for y_ in range(int(y*downscale_ratio), int(math.ceil((y+1)*downscale_ratio))):
                low_diff_y = y_+1 - boundary_y[0]
                upper_diff_y = boundary_y[1] - y_ 
                if low_diff_y < 1 :
                    weight *= low_diff_y
                elif upper_diff_y<1:
                    weight *= upper_diff_y

                weighted_sum += int(map_to_extract_from[y_, x_]) * weight
                sum_of_weights += weight

        weighted_avg = weighted_sum / sum_of_weights
        cell_occupied = weighted_avg >= 0.5 
        return cell_occupied
    
    def getNeighbors(self, pos):
        """
        Get the 8 neighboring cells of a cell at position pos in user resolution.
        Arg:
            pos - position of cell in (x, y) in user resolution.
        """
        assert pos[0] > -1 and pos[1] > -1
        neighbors = []

        for i in range(-1, 2):
            for j in range(-1, 2):
                if i==0 and j==0:
                    continue
                try:
                    neighbor = self.is_cell_occupied(pos[0]+j, pos[1]+i)
                    if not neighbor: # ie it does not have an obstacle
                        neighbors.append((pos[0]+j, pos[1]+i))
                except IndexError:
                    continue
        return neighbors

    # Conversion back to world coordinates
    def convert_to_world_coordinates(self, list_of_points):
        """
        Given a list of coordinates in the map at user (current) resolution, convert each coordinate to the environment coordinates.
        """
        assert self.world_mins is not None 
        np_points = np.array(list_of_points)

        # scaling back to raw_map resolution
        upscale = self.init_resolution / self.current_resolution
        np_points_upscaled = np_points * upscale

        # scaling back to world resolution
        np_points_downscaled = np_points_upscaled / self.init_resolution
        # translation
        np_points_translated = np_points_downscaled + np.flip(self.world_mins)

        return [tuple(x.tolist()) for x in np_points_translated] # convert it back to a list of tuples

    # User resolution methods
    def change_resolution(self, new_resolution):
        assert new_resolution <= self.init_resolution
        self.current_resolution = new_resolution

    def _create_user_map(self):
        h = int(self.height*(self.current_resolution/self.init_resolution))
        w = int(self.width*(self.current_resolution/self.init_resolution))
        user_map = np.zeros((h, w), dtype=bool)
        for x in range(w):
            for y in range(h):
                user_map[y, x] = self.is_cell_occupied(x, y)
        return user_map

    # inflation method
    def inflate_obstacles(self, mobile_bases_bounding_box, inflation_scaling_factor=1.0, half_sizes=True): 
        assert len(mobile_bases_bounding_box) == 2

        if half_sizes: # mujoco box type geoms sizes are given as half-length and half-width
            mobile_bases_bounding_box *= 2

        # find diagonal:
        # its called inflation radius because the the structure's (circle's) center is marked as an obstacle.
        # so to get the entire structure to be marked as the obstacle the larger structure must be twice the size of the original structure
        inflation_radius = np.sqrt(mobile_bases_bounding_box[0]**2 + mobile_bases_bounding_box[1]**2)
        inflation_radius *= inflation_scaling_factor

        # create circular structure
        map_resolution_radius = inflation_radius * self.init_resolution
        center = map_resolution_radius/2
        X, Y = np.meshgrid(np.arange(int(map_resolution_radius)), np.arange(int(map_resolution_radius))) # indices of the structure
        dists = np.sqrt((X - center)**2 + (Y - center)**2)
        map_resolution_struct = dists <= map_resolution_radius/2

        # dilate/inflate the raw_map and create another inflated_map
        self.inflated_map = binary_dilation(self.raw_map, structure=map_resolution_struct)


    # Visualization methods
    def visualize(self):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=75)
        user_map = self._create_user_map()
        ax.imshow(user_map, cmap="gray_r", origin="lower", interpolation="nearest")
        ax.set_title("Occupancy Map with Floor Outline")
        ax.set_xlabel("X (grid cells)")
        ax.set_ylabel("Y (grid cells)")
        plt.tight_layout()
        # plt.savefig("fixture_map_with_floor_box.png")

        # ---- Hover tooltip code below ----
        # annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
        #                     bbox=dict(boxstyle="round", fc="w"),
        #                     arrowprops=dict(arrowstyle="->"))
        # annot.set_visible(False)
        
        # def update_annot(rect, name, event):
        #     annot.xy = (event.xdata, event.ydata)
        #     annot.set_text(name)
        #     annot.get_bbox_patch().set_facecolor('yellow')
        #     annot.set_visible(True)
        
        # def hover(event):
        #     vis = annot.get_visible()
        #     if event.inaxes == ax:
        #         for rect in mapper.patches:
        #             contains, _ = rect.contains(event)
        #             if contains:
        #                 name = mapper.patch_to_name[rect]
        #                 update_annot(rect, name, event)
        #                 fig.canvas.draw_idle()
        #                 return
        #     if vis:
        #         annot.set_visible(False)
        #         fig.canvas.draw_idle()
        
        # fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.show()

    def visualize_path(self, path, type="world_coords"):
        """
        Visualize path in current 
        """
        if type == "world_coords":
            path_ = np.array(path)
            path_ = path_ - np.flip(self.world_mins)
            path = path_ * self.init_resolution

        fig, ax = plt.subplots(figsize=(8, 8), dpi=75)
        ax.imshow(self.raw_map, cmap="gray_r", origin="lower", interpolation="nearest")
        ax.scatter(x=[p[0] for p in path], y=[p[1] for p in path], c="red")
        ax.set_title("Occupancy Map with Floor Outline")
        ax.set_xlabel("X (grid cells)")
        ax.set_ylabel("Y (grid cells)")
        plt.tight_layout()
        plt.show()
        pass


    # def smooth_map(self, iterations=2):
        # if self.raw_map is not None:
        #     self.raw_map = binary_closing(self.raw_map, structure=np.ones((3, 3)), iterations=iterations)
        #     self.label_map = np.where(self.raw_map, self.label_map, 0)