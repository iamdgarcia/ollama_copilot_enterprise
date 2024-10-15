import subprocess


def run_manim(file_name: str, scene_name: str):
    """Runs the Manim command and catches errors."""
    try:
        # Manim command to be executed
        command = f"manim -pql {file_name} {scene_name}"
        
        # Run the command and capture output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Check if the command was successful
        if result.returncode != 0:
            print(f"Error running Manim:\n{result.stderr}")
            raise RuntimeError(f"Manim failed with error: {result.stderr}")
        
        print(f"Manim output:\n{result.stdout}")

    except subprocess.CalledProcessError as e:
        print(f"Failed to execute Manim. Error: {e}")
        raise