from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from tkinter import font

import os
from numpy import pi

from utility.world_handler import WorldHandler
from utility.file_io import get_project_root

checkbox_options = ['Random landmarks', 'Random spawn', 'Grid-based landmarks', 'Row obstacles',
                    'Aerial agents', 'Render result', 'Only']
text_options = ['Number of landmarks', 'Number of agents', 'Grid size', 'Grid rotation']
landmark_options = ['From topology', 'Grid', 'Random', 'Random (clustered)']
default_env_name = 'a'


class GUI:
    """Graphical User Interface for the path planning and target assignment tool. Designed using Tkinter."""
    def __init__(self, vrp_solver=None):
        self.kml_dataset_path = os.path.join(get_project_root(), 'top_dataset', 'kml')
        self.dict_path = os.path.join(get_project_root(), 'top_dataset', 'dict')
        self.input_path = os.path.join(self.kml_dataset_path, 'dataset', 'a.kml')
        self.output_path = os.path.join(get_project_root(), 'output')
        self.input_options = {'just_planning': True}
        self.output_options = {}
        self.preload_toggle = True
        self.preloaded = False
        self.vars = {}

        self.world_handler = WorldHandler(vrp_solver)
        self.algs = dict((n, t) for t, n in self.world_handler.solver.names.items())

        # WINDOW ------------------------------------------------------------
        self.window = Tk()

        self.window.title('Path planning tool')
        self.window.resizable(0, 0)  # make window fixed size

        self.nw_frame = Frame(self.window)
        self.ne_frame = Frame(self.window)
        self.se_frame = Frame(self.window)
        self.sw_frame = Frame(self.window)
        self.bl_frame = Frame(self.window)
        self.br_frame = Frame(self.window)

        self.nw_frame.grid(row=0, column=0, sticky=NW)
        self.ne_frame.grid(row=0, column=2, sticky=NW)
        self.se_frame.grid(row=2, column=2, sticky=NW)
        self.sw_frame.grid(row=2, column=0, sticky=NW)
        self.bl_frame.grid(row=4, column=0)
        self.br_frame.grid(row=4, column=2, sticky=NW)

        Separator(self.window, orient=VERTICAL).grid(row=0, column=1, rowspan=3, sticky='ns')
        Separator(self.window, orient=HORIZONTAL).grid(row=1, column=0, columnspan=3, sticky='we')
        Separator(self.window, orient=HORIZONTAL).grid(row=3, column=0, columnspan=3, sticky='we')
        Separator(self.window, orient=HORIZONTAL).grid(row=5, column=0, columnspan=3, sticky='we')

        # div title labels
        font_u = font.Font(font='TkDefaultFont')
        font_u.configure(underline=True)
        Label(self.nw_frame, text='File input', font=font_u).grid(row=0, column=0, sticky=W)
        Label(self.ne_frame, text='Input settings', font=font_u).grid(row=0, column=0, sticky=W)
        Label(self.se_frame, text='Output settings', font=font_u).grid(row=0, column=0, sticky=W)
        Label(self.sw_frame, text='File output', font=font_u).grid(row=0, column=0, sticky=W)

        # kml file loading
        # file loading needs selection between kml or dataset
        self.current_file_display = Text(self.nw_frame, width=50, height=1, background='gray90', wrap=NONE,
                                         state='disabled', font='TkDefaultFont')
        self.current_file_display.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        self.current_file_display.tag_configure('right', justify='right')
        self.display_blocked_text(self.current_file_display, self.input_path)
        Button(self.nw_frame, text='Load .kml file', command=self.open_kml_file).grid(row=2, column=0, padx=(20, 0))
        Button(self.nw_frame, text='Load database instance', command=self.open_dataset_instance).grid(row=2, column=1,
                                                                                                      padx=(0, 20))

        # output file location
        self.output_file_display = Text(self.sw_frame, width=50, height=1, background='gray90', wrap=NONE,
                                        state='disabled', font='TkDefaultFont')
        self.output_file_display.grid(row=1, column=0, padx=10, pady=10)
        self.output_file_display.tag_configure('right', justify='right')
        self.display_blocked_text(self.output_file_display, self.output_path)
        Button(self.sw_frame, text='Select output directory', command=self.set_output_folder).grid(row=2, column=0,
                                                                                                   pady=(0, 10))

        # input options
        Label(self.ne_frame, text='Agent settings').grid(row=1, column=0, padx=(0, 10), pady=(10, 10), sticky=W)
        self.vars['random_spawn_a'] = BooleanVar(value=True)
        self.spawnbox = Checkbutton(self.ne_frame, text='Random spawn location', variable=self.vars['random_spawn_a'])
        self.spawnbox.grid(row=1, column=1)
        self.vars['ground_flag'] = BooleanVar(value=True)
        self.groundbox = Checkbutton(self.ne_frame, text='Ground-based agents', variable=self.vars['ground_flag'])
        self.groundbox.grid(row=1, column=3)
        Label(self.ne_frame, text='Amount: ').grid(row=1, column=4, sticky=E)
        self.agamountbox = Text(self.ne_frame, width=4, height=1, font='TkDefaultFont', wrap=NONE)
        self.agamountbox.grid(row=1, column=5)
        self.agamountbox.insert('1.0', '3')

        Label(self.ne_frame, text='Landmark settings').grid(row=2, column=0, padx=(0, 10), pady=(10, 10), sticky=W)
        Label(self.ne_frame, text='Type: ').grid(row=2, column=2, sticky=E)
        self.vars['lmtype'] = StringVar(value='Random')
        self.lmtypebox = Combobox(self.ne_frame, textvariable=self.vars['lmtype'],
                                  values=landmark_options, state='readonly', width=20)
        self.lmtypebox.bind("<<ComboboxSelected>>", self.toggle_landmarks)
        self.lmtypebox.grid(row=2, column=3)
        Label(self.ne_frame, text='Amount: ').grid(row=2, column=4, sticky=E)
        self.lmamountbox = Text(self.ne_frame, width=4, height=1, font='TkDefaultFont', wrap=NONE)
        self.lmamountbox.grid(row=2, column=5)
        self.lmamountbox.insert('1.0', '10')

        Label(self.ne_frame, text='Grid settings').grid(row=3, column=0, padx=(0, 10), pady=(10, 10), sticky=W)
        self.vars['row_structure'] = BooleanVar()
        self.rowbox = Checkbutton(self.ne_frame, text='Row obstacles', variable=self.vars['row_structure'])
        self.rowbox.grid(row=3, column=1, sticky=W)
        Label(self.ne_frame, text='Grid size: ').grid(row=3, column=2)
        self.gridsizebox = Text(self.ne_frame, width=6, height=1, font='TkDefaultFont', wrap=NONE)
        self.gridsizebox.grid(row=3, column=3, sticky=W)
        self.gridsizebox.insert('1.0', '0.05')
        Label(self.ne_frame, text='Grid rotation (degrees): ').grid(row=3, column=4)
        self.gridrotbox = Text(self.ne_frame, width=4, height=1, font='TkDefaultFont', wrap=NONE)
        self.gridrotbox.grid(row=3, column=5)
        self.gridrotbox.insert('1.0', '0')

        # output options
        self.output_options['render'] = BooleanVar(value=True)
        self.output_options['store_im'] = BooleanVar(value=False)
        self.output_options['store_kml'] = BooleanVar(value=False)
        self.output_options['store_gps'] = BooleanVar(value=True)

        Checkbutton(self.se_frame, text='Render output',
                    variable=self.output_options['render']).grid(row=1, column=0, padx=10, pady=10)
        Checkbutton(self.se_frame, text='Store solution images',
                    variable=self.output_options['store_im']).grid(row=1, column=1, padx=10, pady=10)
        Checkbutton(self.se_frame, text='Store output paths as kml',
                    variable=self.output_options['store_kml']).grid(row=1, column=2, padx=10, pady=10)
        Checkbutton(self.se_frame, text='Store output paths as GPS',
                    variable=self.output_options['store_gps']).grid(row=1, column=3, padx=10, pady=10)

        self.current_alg = StringVar(value='Clarke-Wright')
        Label(self.se_frame, text='Algorithm: ').grid(row=2, column=0, padx=(10, 0), pady=10)
        self.algbox = Combobox(self.se_frame, textvariable=self.current_alg,
                               values=sorted(list(self.algs.keys())), state='readonly', width=26)
        self.algbox.grid(row=2, column=1, padx=(0, 10))

        Label(self.se_frame, text='Environment name: ').grid(row=2, column=2, padx=(10, 0), pady=10)
        self.envnamebox = Text(self.se_frame, width=10, height=1, font='TkDefaultFont', wrap=NONE)
        self.envnamebox.grid(row=2, column=3)
        self.envnamebox.insert('1.0', 'a')

        # start/stop button section
        Button(self.bl_frame, text='Make environment', width=18,
               command=self.create_environment).grid(row=0, column=0, pady=20)
        Button(self.bl_frame, text='Store environment', width=18,
               command=self.store_environment).grid(row=0, column=1, pady=20)
        Button(self.bl_frame, text='Create solution', width=18,
               command=self.create_solution).grid(row=1, column=0, pady=20)
        Button(self.bl_frame, text='Close environment', width=18,
               command=self.close_environment).grid(row=1, column=1, pady=20)

        # log section
        Label(self.br_frame, text='Output log').grid(row=0, column=0, sticky=W, pady=(10, 0))
        self.logbox = Text(self.br_frame, width=80, height=10, state='disabled')
        self.logbox.grid(row=1, column=0, pady=10, padx=10)

        # exit button
        Button(self.window, text='Exit', width=6, command=self.close_all).grid(row=6, column=0, columnspan=4, pady=10)

        self.get_flags()

    def run(self):
        """Start GUI thread process."""
        self.window.mainloop()

    def close_all(self):
        """Closes the GUI and exits the python script."""
        self.world_handler.close_env()
        self.window.destroy()
        exit()

    def close_environment(self):
        """Attempt to close current rendered environment."""
        # TODO add reset for all class variables as well?
        e = self.world_handler.close_env()
        if len(e) > 0:
            self.log_line('Error! ' + str(e))

    def button_placeholder(self):
        """Placeholder function for testing buttons."""
        pass

    def open_kml_file(self):
        """Set .kml input file for creating a new topology."""
        self.input_path = filedialog.askopenfilename(filetypes=[('Google Earth .kml files', '*.kml')],
                                                     initialdir=self.kml_dataset_path)
        self.display_blocked_text(self.current_file_display, self.input_path)
        self.toggle_preloaded(False)

    def open_dataset_instance(self):
        path = filedialog.askopenfilename(filetypes=[('Scenario, topology or world dictionary', '*.dictionary')],
                                          initialdir=self.dict_path)
        self.input_path = path[0:path.rfind('_')]
        self.display_blocked_text(self.current_file_display, self.input_path)
        self.toggle_preloaded(True)

    def set_output_folder(self):
        """Set output folder, both in this class and in world_handler."""
        self.output_path = filedialog.askdirectory(initialdir='.')
        self.display_blocked_text(self.output_file_display, self.output_path)
        self.world_handler.output_path = self.output_path

    def toggle_algbox(self):
        """Disable algorithm selection box."""
        if self.algbox['state'].string != DISABLED:
            self.algbox.configure(state='disabled')
        else:
            self.algbox.configure(state='readonly')

    def toggle_landmarks(self, event):
        """Toggle landmark amount box if type is grid."""
        if self.lmtypebox.get() == 'Grid':
            self.lmamountbox.delete(0.0, END)
            self.lmamountbox.insert(END, 'N/A')
            self.lmamountbox.configure(state='disabled', background='gray90')
        else:
            self.lmamountbox.configure(state='normal', background='white')
            self.set_text_value(self.lmamountbox, self.world_handler.default_flags['landmark_amount'])

    def toggle_preloaded(self, preloaded):
        """Toggle preloaded flag for relevant other settings."""
        self.preloaded = preloaded
        # if preloaded and self.preload_toggle:
        #     self.preload_toggle = False  # toggle off
        #     self.preloaded = True
        #
        # elif not preloaded and not self.preload_toggle:
        #     self.preload_toggle = True
        #     self.preloaded = False

    def log_line(self, text):
        """Print text to the logger on a new line."""
        self.logbox.configure(state='normal')
        self.logbox.insert(END, text + '\n')
        self.logbox.configure(state='disabled')
        self.logbox.yview(END)

    def get_flags(self):
        """Get all input options from their respective fields and transfer them to the input flags."""
        self.input_options['from_grid_l'] = True if self.vars['lmtype'].get() == 'Grid' else False
        self.input_options['cluster_l'] = True if self.vars['lmtype'].get() == 'Random (clustered)' else False
        self.input_options['random_spawn_l'] = True if self.vars['lmtype'].get().find('Random') != -1 else False

        self.input_options['random_spawn_a'] = self.vars['random_spawn_a'].get()
        self.input_options['row_structure'] = self.vars['row_structure'].get()
        self.input_options['ground_flag'] = self.vars['ground_flag'].get()

        try:
            self.input_options['landmark_amount'] = int(self.get_value(self.lmamountbox))
        except ValueError:
            if self.get_value(self.lmamountbox) == 'N/A':
                self.input_options['landmark_amount'] = self.world_handler.default_flags['landmark_amount']
            else:
                self.set_text_value(self.lmamountbox, str(self.world_handler.default_flags['landmark_amount']))
                self.input_options['landmark_amount'] = int(self.get_value(self.lmamountbox))

        try:
            self.input_options['agent_amount'] = int(self.get_value(self.agamountbox))
        except ValueError:
            self.set_text_value(self.agamountbox, str(self.world_handler.default_flags['agent_amount']))
            self.input_options['agent_amount'] = int(self.get_value(self.agamountbox))

        try:
            self.input_options['grid_size_l'] = float(self.get_value(self.gridsizebox))
        except ValueError:
            self.set_text_value(self.gridsizebox, str(self.world_handler.default_flags['grid_size_l']))
            self.input_options['grid_size_l'] = float(self.get_value(self.gridsizebox))

        try:
            self.input_options['grid_rotation'] = float(self.get_value(self.gridrotbox)) * pi / 180
        except ValueError:
            self.set_text_value(self.gridrotbox, str(self.world_handler.default_flags['grid_rotation'] * 180 / pi))
            self.input_options['grid_rotation'] = float(self.get_value(self.gridrotbox)) * pi / 180

    def set_flags(self, flags):
        """Set flag fields to a specific value."""
        if flags['from_grid_l']:
            self.vars['lmtype'].set('Grid')
        elif flags.get('clustered', False):
            self.vars['lmtype'].set('Random (clustered)')
        elif flags['random_spawn_l'] and not flags.get('clustered', False):
            self.vars['lmtype'].set('Random')
        else:
            self.vars['lmtype'].set('From topology')

        self.vars['random_spawn_a'].set(flags['random_spawn_a'])
        self.vars['row_structure'].set(flags['row_structure'])
        self.vars['ground_flag'].set(flags['ground_flag'])

        self.set_text_value(self.lmamountbox, str(flags['landmark_amount']))
        self.set_text_value(self.agamountbox, str(flags['agent_amount']))
        self.set_text_value(self.gridsizebox, str(flags['grid_size_l']))
        self.set_text_value(self.gridrotbox, str(flags['grid_rotation'] * 180 / pi))

    def create_environment(self):
        """Creates new environment, either from preloaded class files or from a new .kml topology."""
        self.close_environment()
        self.get_flags()

        try:
            self.world_handler.create_environment(self.preloaded, self.input_path,
                                                  self.output_options['render'].get(), **self.input_options)
            self.log_line('Environment setup complete.')

            if self.preloaded:
                self.set_flags(self.world_handler.get_flags())
                self.set_text_value(self.envnamebox, os.path.basename(os.path.normpath(self.input_path)))
        except Exception as e:
            self.log_line('Error! ' + str(e))

    def store_environment(self):
        """Saves class variables."""
        env_tag = self.get_value(self.envnamebox)
        env_tag = default_env_name if len(env_tag) == 0 else env_tag
        self.world_handler.store_objects(os.path.join(self.output_path, env_tag))

    def create_solution(self):
        """Attempt to solve the environment path planning problem, using the set output options."""
        alg = self.algs[self.current_alg.get()]
        if not self.world_handler.check_loaded():
            self.log_line('Unable to create solution. Environment not loaded.')
            return
        else:
            self.log_line('Creating solution for ' + alg + '...')

        try:
            self.world_handler.apply_algorithm(alg, self.output_options['store_im'].get(),
                                               self.output_options['store_kml'].get(),
                                               self.output_options['store_gps'].get())
            self.log_line('Solution found.')
        except Exception as e:
            self.log_line('Error! ' + str(e))

    @staticmethod
    def get_value(field):
        """Gets the current value of a text box."""
        return field.get(1.0, 'end-1c')

    @staticmethod
    def set_text_value(field, text):
        """Sets a text box to a value."""
        field.delete(0.0, END)
        field.insert(END, text)

    @staticmethod
    def display_blocked_text(field, text):
        """Sets a disabled text box to a specific value."""
        field.configure(state='normal')
        field.delete(0.0, END)
        field.insert(END, text)
        field.tag_add('right', '1.0', END)
        field.configure(state='disabled')


if __name__ == '__main__':
    gui = GUI()
    gui.run()
