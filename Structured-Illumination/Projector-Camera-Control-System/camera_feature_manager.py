import datetime  # For timestamp in feature file
import traceback  # For printing exception details


class CameraFeatureManager:
    """
    Manager for reading and updating camera features.
    """

    def __init__(self, camera):
        # Store the camera object for feature operations
        self.camera = camera

    def read_features(self, output_file: str) -> dict:
        """
        Read all camera features, write them to a file, and return a feature dict.
        :param output_file: Path to save the feature report
        :return: Dictionary of feature metadata and values
        """
        # Retrieve all feature objects from camera
        features = self.camera.get_all_features()
        features_dict = {}
        # Prepare file header with camera ID and current time
        header = (
            f"Camera Features for camera: {self.camera.get_id()}\n"
            f"Time: {datetime.datetime.now()}\n" + "=" * 50 + "\n\n"
        )
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                # Write header to file
                f.write(header)
                # Iterate through each feature
                for feat in features:
                    try:
                        name = feat.get_name()  # Feature identifier
                        display_name = feat.get_display_name()  # Human-readable name
                        tooltip = feat.get_tooltip()  # UI tooltip text
                        description = feat.get_description()  # Detailed description
                        # Read current value if readable
                        if feat.is_readable():
                            value = feat.get()
                            value_type = type(value).__name__
                        else:
                            value = "Not Readable"
                            value_type = "Unknown"
                        # Store metadata in dictionary
                        features_dict[name] = {
                            "Display Name": display_name,
                            "Tooltip": tooltip,
                            "Description": description,
                            "Current Value": value,
                            "Value DataType": value_type,
                        }
                        # Write feature details to report file
                        f.write(f"Feature Name   : {name}\n")
                        f.write(f"Display Name   : {display_name}\n")
                        f.write(f"Tooltip        : {tooltip}\n")
                        f.write(f"Description    : {description}\n")
                        f.write(f"Current Value  : {value}\n")
                        f.write(f"Value DataType : {value_type}\n")
                        f.write("-" * 50 + "\n")
                    except Exception:
                        # Print traceback and log error in file
                        traceback.print_exc()
                        f.write("Error reading feature.\n" + "-" * 50 + "\n")
                        # Ensure dictionary has an entry for unknown
                        features_dict["Unknown"] = {"Error": "Error reading feature."}
            print("Features information saved to", output_file)
        except Exception:
            # Handle file I/O errors
            traceback.print_exc()
        return features_dict

    def update_features(self, features_to_update: dict, output_file: str):
        """
        Update camera features based on provided dict, then refresh report.
        :param features_to_update: Mapping feature names to new values
        :param output_file: Path to rewrite the feature report
        :return: (updated_features_dict, success_count)
        """
        success_count = 0
        # Iterate through features to be updated
        for feat_name, new_value in features_to_update.items():
            try:
                feat = self.camera.get_feature_by_name(feat_name)  # Get feature object
            except Exception as e:
                # Log if feature not found
                print(f"Could not find feature '{feat_name}': {e}")
                traceback.print_exc()
                continue
            # Skip if feature is not writable
            if not feat.is_writeable():
                print(f"Feature '{feat_name}' is not writable.")
                continue
            try:
                feat.set(new_value)  # Apply new value
                success_count += 1
                print(f"Feature '{feat_name}' updated to {feat.get()} successfully.")
            except Exception as e:
                # On failure, attempt to read current value
                try:
                    current_val = feat.get()
                except Exception:
                    current_val = "Unavailable"
                print(
                    f"Feature '{feat_name}' update failed: {e}. Current value is {current_val}."
                )
                traceback.print_exc()
        # After updates, rewrite the feature report
        print("All features have been updated. Refreshing feature file…")
        updated_features = self.read_features(output_file)
        return updated_features, success_count
