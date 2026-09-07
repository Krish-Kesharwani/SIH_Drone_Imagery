import os


def save_ply(points, filename):

    os.makedirs("outputs", exist_ok=True)

    filepath = f"outputs/{filename}"

    with open(filepath, "w") as f:

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")

        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        f.write("end_header\n")

        for p in points:
            f.write(
                f"{p[0]} {p[1]} {p[2]}\n"
            )

    return filepath