import numpy as np
# from scipy.ndimage import binary_closing

# for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


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

    def __init__(self, type="grid", init_resolution=100):
        self.map = None
        self.type = type
        self.init_resolution = self.current_resolution = init_resolution

    def set_size(self, height, width):
        assert self.type == "grid"
        self.map = np.zeros((height, width), dtype=bool)
        self.height = height
        self.width = width

    def set_rect_obstacles(self, x0, y0, x1, y1):
        """
        Set a rectangular obstacle on the map at the init resolution.
        """
        assert self.current_resolution == self.init_resolution
        self.map[y0:y1, x0:x1] = True

    def change_resolution(self, new_resolution):
        assert new_resolution <= self.init_resolution
        self.current_resolution = new_resolution
   
    def is_cell_occupied(self, x, y):
        """
        Returns whether the cell at the user resolution is occupied or not.
        """
        if x < 0 or y < 0 or x >= self.width*(self.current_resolution/self.init_resolution) or y >= self.height*(self.current_resolution/self.init_resolution):
            raise IndexError

        if self.current_resolution == self.init_resolution:
            return self.map[y, x]
        

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

                weighted_sum += int(self.map[y_, x_]) * weight
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

    def convert_to_world_coordinates(self, list_of_points):
        """
        Given a list of coordinates in the map, convert each coordinate to the environment coordinates.
        """
        np_points = np.array(list_of_points)
        upscale = self.init_resolution / self.current_resolution
        np_points_upscaled = np_points * upscale
        return [tuple(x.tolist()) for x in np_points_upscaled] # convert it back to a list of tuples

    def _create_user_map(self):
        h = int(self.height*(self.current_resolution/self.init_resolution))
        w = int(self.width*(self.current_resolution/self.init_resolution))
        user_map = np.zeros((h, w), dtype=bool)
        for x in range(w):
            for y in range(h):
                user_map[y, x] = self.is_cell_occupied(x, y)
        return user_map
    
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

    def visualize_path(self, path):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=75)
        ax.imshow(self.map, cmap="gray_r", origin="lower", interpolation="nearest")
        ax.scatter(x=[p[0] for p in path], y=[p[1] for p in path], c="red")
        ax.set_title("Occupancy Map with Floor Outline")
        ax.set_xlabel("X (grid cells)")
        ax.set_ylabel("Y (grid cells)")
        plt.tight_layout()
        plt.show()
        pass


    # def smooth_map(self, iterations=2):
        # if self.map is not None:
        #     self.map = binary_closing(self.map, structure=np.ones((3, 3)), iterations=iterations)
        #     self.label_map = np.where(self.map, self.label_map, 0)