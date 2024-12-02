import tkinter as tk
import numpy as np

class GomokuGUI:
    def __init__(self, board_size):
        self.board_size = board_size  # size of board, this will be a grid of (board_size-1 x board_size-1)
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.player = 1  # 1: black, 2: white
        
        self.load_gui()
    

    def load_gui(self):
        self.interface = tk.Tk()
        self.interface.title("Gomoku")

        self.cell_size = 60  # Size of each cell
        self.width, self.height = self.cell_size * self.board_size, self.cell_size * self.board_size
        Frame_Gap = 100  # Gap between the canvas and the window frame

        self.margin = 40 # Margin around the grid (for labels)
        self.box_padding = 15  # Padding between the grid and the outer box
        self.total_padding = self.margin + self.box_padding
        self.canvas_size = self.board_size * self.cell_size
        self.total_size = self.canvas_size + 2 * self.total_padding - self.cell_size

        self.canvas = tk.Canvas(self.interface, width=self.total_size,
                                height=self.total_size, background="tan3")
        self.canvas.pack()  # Pack the canvas into the window
        self.interface.geometry(f"{self.total_size}x{self.total_size}")

        # Draw the grid for the board
        for i in range(self.board_size):
            # Vertical lines (grid columns), starting and ending at margin
            self.canvas.create_line(self.total_padding + self.cell_size * i, self.total_padding, 
                                    self.total_padding + self.cell_size * i, self.canvas_size + self.total_padding - self.cell_size, fill="black")
            # Horizontal lines (grid rows), starting and ending at margin
            self.canvas.create_line(self.total_padding, self.total_padding + self.cell_size * i, 
                                    self.canvas_size + self.total_padding - self.cell_size, self.total_padding + self.cell_size * i, fill="black")
            
            # Label the rows on the left side
            self.canvas.create_text(self.margin / 2, self.total_padding + self.cell_size * i, 
                                    text=str(i + 1), font="Helvetica 12 bold", anchor="center")
            # Label the columns above the grid
            self.canvas.create_text(self.total_padding + self.cell_size * i, self.margin / 2, 
                                    text=str(i + 1), font="Helvetica 12 bold", anchor="center")
        
        # Draw the outer box around the grid with extra padding
        self.canvas.create_rectangle(self.total_padding - self.box_padding, 
                                    self.total_padding - self.box_padding, 
                                    self.canvas_size + self.total_padding + self.box_padding - self.cell_size, 
                                    self.canvas_size + self.total_padding + self.box_padding - self.cell_size, 
                                    width=3, outline="black")

        self.canvas.bind("<Button-1>", self.handle_click)
        
    # Function handle_click(event):
    #     - Get the mouse position from the event
    #     - Translate the mouse position to board coordinates (row, col)
    #     - Check if the action is valid (e.g., empty cell)
    #     - Call the GomokuSimulator step() function with the clicked position
    #     - If valid, update the game board and switch players
    #     - If invalid, display an error message or handle accordingly   
    def handle_click(self, event):
        x, y = event.x, event.y
        
        row, col = self.get_board_position(x, y)  # Get the board position from the mouse click

        print(f"Clicked on row {row}, col {col}")

        if self.board[row, col] == 0:
            self.board[row, col] = self.player
            self.draw_piece(row, col)
            if self.check_win(row, col):
                self.display_winner()
            elif np.all(self.board != 0):
                print("It's a draw!")
            else:
                self.switch_player()
        else:
            print("Invalid move, cell is already occupied")
        
        
    
    def get_board_position(self, x, y):
        row = (y - self.total_padding) // self.cell_size
        col = (x - self.total_padding) // self.cell_size
        return row, col


    def display_winner(self):
        winner = "Black" if self.player == 1 else "White"
        self.canvas.create_text(self.total_size / 2, self.total_size - self.margin / 2, 
                        text=f"Player {winner} wins!", font="Helvetica 14 bold", anchor="center", fill="black")

    def check_horizontal(self, row, col):
        c = 0
        for j in range(self.board_size):
            if self.board[row, j] == self.player:
                c += 1
                if c == 5:
                    return True
            else:
                c = 0
        return False
    
    def check_vertical(self, row, col):
        c = 0
        for i in range(self.board_size):
            if self.board[i, col] == self.player:
                c += 1
                if c == 5:
                    return True
            else:
                c = 0
        return False

    def check_diagonal(self, row, col):
        c = 0
        # Check diagonal from top-left to bottom-right
        for i in range(-4, 5):
            if 0 <= row + i < self.board_size and 0 <= col + i < self.board_size:
                if self.board[row + i, col + i] == self.player:
                    c += 1
                    if c == 5:
                        return True
                else:
                    c = 0
        # Check diagonal from top-right to bottom-left
        c = 0
        for i in range(-4, 5):
            if 0 <= row + i < self.board_size and 0 <= col - i < self.board_size:
                if self.board[row + i, col - i] == self.player:
                    c += 1
                    if c == 5:
                        return True
                else:
                    c = 0
        return False

    def check_win(self, row, col):
        """Check if the player just won with their most recent move."""
        if self.check_horizontal(row, col) or self.check_vertical(row, col) or self.check_diagonal(row, col):
            return True
        return False
 

    def draw_piece(self, row, col):
        # Needs to draw on the intersection of the lines
        x_pos = self.total_padding + col * self.cell_size
        y_pos = self.total_padding + row * self.cell_size
        color = "black" if self.player == 1 else "white"
        radius = self.cell_size // 3

        self.canvas.create_oval(x_pos - radius, y_pos - radius, x_pos + radius, y_pos + radius, fill=color, outline="")

    def switch_player(self):
        self.player = 1 if self.player == 2 else 2

    def run(self):
        self.interface.mainloop()

def main():
    game = GomokuGUI(9)
    game.run()


if __name__ == '__main__':
    main()
