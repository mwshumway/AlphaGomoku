"""Gomoku.py - A document that imlpements the Gomoku game."""

import tkinter as tk
import numpy as np


class Gomoku:
    def __init__(self, board_size, gui=True):
        self.board_size = board_size
        self.board = np.zeros((board_size + 1, board_size + 1), dtype=int)
        self.gui = gui

        if self.gui:
            self.cell_size = 60 # Size of each cell
            self.width, self.height = self.cell_size * self.board_size, self.cell_size * self.board_size
            Frame_Gap = 100  # Gap between the canvas and the window frame
            self.interface = tk.Tk()  # Create the main window
            self.interface.geometry(f"{self.width}x{self.height + Frame_Gap}")
            self.setup_gui()
    
    def setup_gui(self):
        self.interface.title("Gomoku")  # Set the title of the window
        self.margin = 40  # Margin around the grid (for labels)
        self.box_padding = 15  # Padding between the grid and the outer box
        
        # Adjust canvas size to include margin and padding for the outer box
        total_padding = self.margin + self.box_padding
        canvas_size = self.board_size * self.cell_size  # Size of the grid itself (without margins)
        total_canvas_size = canvas_size + 2 * total_padding  # Size of the grid + margins

        self.canvas = tk.Canvas(self.interface, 
                                width=canvas_size + 2 * total_padding, 
                                height=canvas_size + 2 * total_padding, 
                                background="tan3")
        self.canvas.pack()  # Pack the canvas into the window
        
        # Adjust window size to fit canvas
        self.interface.geometry(f"{canvas_size + 2 * total_padding}x{canvas_size + 2 * total_padding}")

        # Draw the grid for the board
        for i in range(self.board_size + 1):
            # Vertical lines (grid columns), starting and ending at margin
            self.canvas.create_line(total_padding + self.cell_size * i, total_padding, 
                                    total_padding + self.cell_size * i, canvas_size + total_padding, fill="black")
            # Horizontal lines (grid rows), starting and ending at margin
            self.canvas.create_line(total_padding, total_padding + self.cell_size * i, 
                                    canvas_size + total_padding, total_padding + self.cell_size * i, fill="black")
            
            # Label the rows on the left side
            self.canvas.create_text(self.margin / 2, total_padding + self.cell_size * i, 
                                    text=str(i + 1), font="Helvetica 12 bold", anchor="center")
            # Label the columns above the grid
            self.canvas.create_text(total_padding + self.cell_size * i, self.margin / 2, 
                                    text=str(i + 1), font="Helvetica 12 bold", anchor="center")
        
        # Draw the outer box around the grid with extra padding
        self.canvas.create_rectangle(total_padding - self.box_padding, 
                                    total_padding - self.box_padding, 
                                    canvas_size + total_padding + self.box_padding, 
                                    canvas_size + total_padding + self.box_padding, 
                                    width=3, outline="black")
        
        # Add text to the bottom of the canvas
        self.canvas.create_text(total_canvas_size / 2, total_canvas_size - self.margin / 2, 
                        text="Welcome to Gomoku! Begin by clicking on an empty cell.",
                        font="Helvetica 14 bold", anchor="center", fill="black")


        # Bind mouse click event to handle where the user clicks
        self.canvas.bind("<Button-1>", self.handle_click)
    
    def handle_click(self, event):
        # Calculate the closest intersection to the click
        total_padding = 40 + 15  # Margin + box_padding
        x_click = event.x - total_padding
        y_click = event.y - total_padding
        
        # Find the closest intersection based on the grid cell size
        nearest_col = round(x_click / self.cell_size)
        nearest_row = round(y_click / self.cell_size)

        # Update the board with the new piece
        self.board[nearest_row, nearest_col] = 1
        
        # Convert back to canvas coordinates to place the piece at the intersection
        x_pos = total_padding + nearest_col * self.cell_size
        y_pos = total_padding + nearest_row * self.cell_size

        # Draw the game piece at the calculated position
        self.draw_piece(x_pos, y_pos)

    def draw_piece(self, x, y, piece="black"):
        radius = self.cell_size // 3  # Adjust the size of the piece to fit within the grid intersection
        if piece == "black":
            color = "black"
        else:
            color = "white"
        
        # Draw the piece (centered at x, y)
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="")


    def start(self):
        """Start the Tkinter main loop."""
        self.interface.mainloop()

    



    



def main():
    game = Gomoku(9)
    game.start()

    pass





if __name__ == '__main__':
    main()