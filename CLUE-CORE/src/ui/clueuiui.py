#!/usr/bin/python3
import pathlib
import tkinter as tk
import pygubu

PROJECT_PATH = pathlib.Path(__file__).parent
PROJECT_UI = PROJECT_PATH / "CLUE-GUI.ui"
RESOURCE_PATHS = [PROJECT_PATH]


class ClueGuiUI:
    def __init__(
        self,
        master=None,
        translator=None,
        on_first_object_cb=None,
        data_pool=None
    ):
        self.builder = pygubu.Builder(
            translator=translator,
            on_first_object=on_first_object_cb,
            data_pool=data_pool
        )
        self.builder.add_resource_paths(RESOURCE_PATHS)
        self.builder.add_from_file(PROJECT_UI)
        # Main widget
        self.mainwindow: tk.Toplevel = self.builder.get_object(
            "Toplevel6", master)
        # Main menu
        _main_menu = self.builder.get_object("top_menu", self.mainwindow)
        self.mainwindow.configure(menu=_main_menu)
        self.builder.connect_callbacks(self)

    def center_window(self):
        if self.mainwindow.winfo_ismapped():
            min_w, min_h = self.mainwindow.wm_minsize()
            max_w, max_h = self.mainwindow.wm_maxsize()
            screen_w = self.mainwindow.winfo_screenwidth()
            screen_h = self.mainwindow.winfo_screenheight()
            final_w = min(
                screen_w,
                max_w,
                max(
                    min_w,
                    self.mainwindow.winfo_width(),
                    self.mainwindow.winfo_reqwidth(),
                ),
            )
            final_h = min(
                screen_h,
                max_h,
                max(
                    min_h,
                    self.mainwindow.winfo_height(),
                    self.mainwindow.winfo_reqheight(),
                ),
            )
            x = (screen_w // 2) - (final_w // 2)
            y = (screen_h // 2) - (final_h // 2)
            geometry = f"{final_w}x{final_h}+{x}+{y}"

            def set_geometry():
                self.mainwindow.geometry(geometry)

            self.mainwindow.after_idle(set_geometry)
        else:
            # Window is not mapped, wait and try again later.
            self.mainwindow.after(5, self.center_window)

    def run(self, center=False):
        if center:
            self.center_window()
        self.mainwindow.mainloop()


if __name__ == "__main__":
    app = ClueGuiUI()
    app.run()
