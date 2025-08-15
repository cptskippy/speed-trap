import logging
import json

logger = logging.getLogger(__name__)

class SummaryGenerator:
    """
    Helper class that reads a data file and generates summary data.

    Functions:
        generate_summary_file: Processes a data file to generate summary data.
        generate_summary_files: Processes data files to generate summary data.

    """
    def __init__(self):
        """
        Initialize a SummaryGenerator object.

        Args:
            None
        """


    def generate_summary_files(self, data_files):
        for files in data_files:
            data_file = files["data"]
            summary_file = files["summary"]

            logger.info(f"Processing File: {data_file}")
            self.generate_summary_file(data_file, summary_file)

    def generate_summary_file(self, data_file, summary_file):
        
        top_speed, occurred, approaching = self._parse_data_file(data_file)
        license_plate = "" # placeholder for later...

        summary_data = {"top_speed": top_speed,
                        "occurred": occurred,
                        "approaching": approaching,
                        "license_plate": license_plate}
        
        logger.info(f"  Summary Data: {summary_data}")
        self._save_summary_file(summary_file, summary_data)

    def _parse_data_file(self, file_path: str):
        """Finds the top speed in the file that has approaching and retreating speeds"""

        top_speed = ats = rts = 0
        occurred = ao = ro = ""
        approaching = True

        logger.info(f"  Opening file: {file_path}")
        with open(file_path, "r") as f:
            speed_data = json.load(f)

        if speed_data: # Check if list is not empty
            a = speed_data["approaching"]
            if a: # Check if list is not empty
                logger.info("    Finding Max Approaching...")
                ats, ao = self._get_top_speed(a)
            else:
                logger.info(f"    No Approaching data...")

            r = speed_data["retreating"]
            if r: # Check if list is not empty
                logger.info("    Finding Max Retreating...")
                rts, ro = self._get_top_speed(r)
            else:
                logger.info(f"    No Retreating data...")
        else:
            logger.info(f"    File empty: {file_path}")

        if rts > ats: # Reteating
            top_speed = rts
            occurred = ro
            approaching = False
        else:
            top_speed = ats
            occurred = ao
            approaching = True

        return top_speed, occurred, approaching

    def _get_top_speed(self, json_data):
        top_speed = 0.0
        occurred = ""

        if json_data:  # Check if list is not empty
            top_point = max(json_data, key=lambda x: x["speed"])
            top_speed = top_point["speed"]
            occurred = top_point["occurred"]

        return top_speed, occurred

    def _save_summary_file(self, file_path, data):
        """Saves summary data to file"""

        # Save data to disk
        logger.info(f"  Save file: {file_path}")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            logger.info("    File saved")

