import random

# Keep the valid choices in one list so both the computer and input validation use the same options
GameOptions = ["rock", "paper", "scissors"]

# Randomly choose the opponent move from the same valid game options
opponent = random.choice(GameOptions)

player = input("Choose rock, paper, scissors: ")

# Reject invalid input before comparing the player choice with the opponent
if player not in GameOptions:
    print("Invalid choice. Please choose rock, paper, scissors.")

else:
    print(f"\nYou chose: {player}")
    print(f"Opponent chose: {opponent}")

    # Matching choices produce a tie before any win conditions need to be checked
    if player == opponent:
        print("It's a tie!")

    # Each winning combination is listed explicitly so the game rules are easy to follow
    elif (
        (player == "rock" and opponent == "scissors") or
        (player == "paper" and opponent == "rock") or
        (player == "scissors" and opponent == "paper")
    ):
        print("You win!")

    else:
        print("Opponent wins!")