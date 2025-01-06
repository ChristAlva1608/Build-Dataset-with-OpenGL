def imgui_menu(self):
        ########################################################################
        #                           SGD Visualization                          #
        ########################################################################
        
        imgui.new_frame()
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(300, 200)
        imgui.begin("Controls")

        imgui.set_next_item_width(100)
        _, self.selected_obj = imgui.combo(
            "Select Object",
            int(self.selected_obj),
            ["Wuson", "Porsche", "Bathroom", "Building",
             "Castelia City", "House Interior"]
        )

        imgui.set_next_item_width(100)
        if imgui.begin_combo("Select Option", "Options"):
            # Add checkboxes inside the combo
            _, self.single_camera_option = imgui.checkbox("Single Camera", self.single_camera_option)
            _, self.multi_camera_option = imgui.checkbox("Multi Camera", self.multi_camera_option)
            imgui.end_combo()

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(300, 200)
        # imgui.set_next_item_width(100)
        if imgui.begin_combo("","Light"):
        # Add Diffuse slider
        imgui.set_next_item_width(100)
        self.diffuse_changed, diffuse_value = imgui.slider_float("Diffuse", 
                                          self.diffuse, 
                                          min_value=0.00, 
                                          max_value=1,
                                          format="%.2f")
        if self.diffuse_changed:
            self.diffuse = diffuse_value
    
        # Add Ambient slider
        imgui.set_next_item_width(100)
        self.ambient_changed, ambient_value = imgui.slider_float("Ambient", 
                                          self.ambient, 
                                          min_value=0.00, 
                                          max_value=1,
                                          format="%.2f")
        if self.ambient_changed:
            self.ambient = ambient_value

        # Add Specular slider
        imgui.set_next_item_width(100)
        self.specular_changed, specular_value = imgui.slider_float("Specular", 
                                          self.specular, 
                                          min_value=0.00, 
                                          max_value=1,
                                          format="%.2f")
        if self.specular_changed:
            self.specular = specular_value

        if self.multi_camera_option:
            imgui.set_next_item_width(100)
            _, self.selected_vcamera = imgui.combo(
                "Select VCamera",
                int(self.selected_vcamera),
                [str(i) for i in range(1, self.num_vcameras + 1)]
            )

        imgui.set_next_item_width(100)
        if imgui.button("Load Model"):
            self.create_model()

        imgui.set_next_item_width(100)
        if imgui.button("Save all"):
            self.save_rgb()
            self.save_depth()
        
        imgui.same_line()
        imgui.set_next_item_width(100)
        if imgui.button("Save RGB"):
            self.save_rgb()

        imgui.same_line()
        imgui.set_next_item_width(100)
        if imgui.button("Save Depth Map"):
            self.save_depth()

        imgui.end()
        imgui.render()
        self.imgui_impl.render(imgui.get_draw_data())


# Hover and click text
def init_ui(self):
        # Initialize item for "Scene"
        self.scene_items = [
            {"text": f"Clickable Text Item", "function": self.printT, "hovered": False, "clicked": False}
        ]
        
for idx, item in enumerate(self.scene_items):
            if imgui.selectable(item["text"], False)[0]:
                item["function"]()

            if imgui.selectable(item["text"], item["clicked"])[0]:
                item["clicked"] = not item["clicked"]
            
            # Hover detection
            if imgui.is_item_hovered():
                if not item["hovered"]:
                    item["hovered"] = True
                    # Change text color when hovered
                    imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.0, 0.0)
                    imgui.text(f"Hovering over: {item['text']}")
                    imgui.pop_style_color()
            else:
                item["hovered"] = False