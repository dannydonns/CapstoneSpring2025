########################################################################
## SPINN DESIGN CODE
# YOUTUBE: (SPINN TV) https://www.youtube.com/spinnTv
# WEBSITE: spinndesign.com
########################################################################
## IMPORTS
########################################################################
from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.core.window import Window
########################################################################
## SET WINDOW SIZE
########################################################################
Window.size = (350, 600)

########################################################################
## HOT RELOADER CONFIG
########################################################################
KV = '''
#:import KivyLexer kivy.extras.highlight.KivyLexer
#:import HotReloadViewer kivymd.utils.hot_reload_viewer.HotReloadViewer


BoxLayout:
    HotReloadViewer:
        path: app.path_to_kv_file
        errors: True
        errors_text_color: 0, 0, 0, 1
        errors_background_color: app.theme_cls.bg_dark
'''
########################################################################
## MAIN CLASS
########################################################################
class Example(MDApp):
    # Kv file to hot reload
    path_to_kv_file = "mainScreen.kv"
    # Build function
    def build(self):
        return Builder.load_string(KV)
########################################################################
## RUN APP
########################################################################
Example().run()
########################################################################
## END =>
########################################################################








































