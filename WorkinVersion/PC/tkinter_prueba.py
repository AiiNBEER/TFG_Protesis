import tkinter as tk

# Crear la ventana principal
root = tk.Tk()
root.title("Dibujar una Mano")
root.geometry("400x400")

# Crear un lienzo
canvas = tk.Canvas(root, width=400, height=400)
canvas.pack()

# Dibujar la palma de la mano
canvas.create_oval(150, 200, 250, 300, outline="black", fill="peachpuff")

# Dibujar los dedos
# Pulgar
canvas.create_line(200, 250, 130, 170, width=10, fill="peachpuff")
canvas.create_oval(120, 160, 140, 180, outline="black", fill="peachpuff")

# Índice
canvas.create_line(200, 200, 180, 100, width=10, fill="peachpuff")
canvas.create_oval(170, 90, 190, 110, outline="black", fill="peachpuff")

# Medio
canvas.create_line(200, 190, 200, 90, width=10, fill="peachpuff")
canvas.create_oval(190, 80, 210, 100, outline="black", fill="peachpuff")

# Anular
canvas.create_line(200, 200, 220, 100, width=10, fill="peachpuff")
canvas.create_oval(210, 90, 230, 110, outline="black", fill="peachpuff")

# Meñique
canvas.create_line(200, 250, 270, 170, width=10, fill="peachpuff")
canvas.create_oval(260, 160, 280, 180, outline="black", fill="peachpuff")

# Iniciar el bucle principal de la GUI
root.mainloop()