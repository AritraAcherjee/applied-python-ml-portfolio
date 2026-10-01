import random


# The class creates the secret number and prepares the different hint types used during the game
class NumberGuesser:
    def __init__(self, minimum=1, maximum=10):
        self.minimum = minimum
        self.maximum = maximum
        # Choose a new secret number within the allowed range for each game
        self.secret_number = random.randint(minimum, maximum)

        self.divisors = self.__find_divisors()
        self.multiples_list = self.__find_multiples()
        self.lower_hint = self.__choose_lower_number()
        self.higher_hint = self.__choose_higher_number()
        self.even_or_odd = self.__check_even_or_odd()

    # Divisors provide one type of mathematical clue without revealing the number directly
    def __find_divisors(self):
        divisors = []

        for number in range(1, self.secret_number + 1):
            if self.secret_number % number == 0:
                divisors.append(number)

        return divisors

    # Store a few multiples so the player can also receive multiplication-based hints
    def __find_multiples(self):
        multiples_list = []

        for number in range(2, 6):
            multiples_list.append(self.secret_number * number)

        return multiples_list

    # Pick a valid lower bound that can be used for a greater-than hint
    def __choose_lower_number(self):
        if self.secret_number > self.minimum:
            return random.randint(
                self.minimum,
                self.secret_number - 1
            )

        return None

    # Pick a valid upper bound that can be used for a less-than hint
    def __choose_higher_number(self):
        if self.secret_number < self.maximum:
            return random.randint(
                self.secret_number + 1,
                self.maximum
            )

        return None

    # Parity gives the player another clue that is independent of the range hints
    def __check_even_or_odd(self):
        if self.secret_number % 2 == 0:
            return "even"

        return "odd"

    # Randomly vary the hint category so repeated hints are not always the same type
    def give_hint(self):
        hint_categories = ["math", "range", "parity"]
        selected_category = random.choice(hint_categories)

        if selected_category == "math":
            math_options = ["divisor", "multiple"]
            selected_option = random.choice(math_options)

            if selected_option == "divisor":
                divisor = random.choice(self.divisors)
                return f"A divisor of the secret number is {divisor}."

            multiple = random.choice(self.multiples_list)
            return f"A multiple of the secret number is {multiple}."

        elif selected_category == "range":
            range_options = []

            if self.lower_hint is not None:
                range_options.append("lower")

            if self.higher_hint is not None:
                range_options.append("higher")

            selected_option = random.choice(range_options)

            if selected_option == "lower":
                return (
                    f"The secret number is greater than "
                    f"{self.lower_hint}."
                )

            return (
                f"The secret number is less than "
                f"{self.higher_hint}."
            )

        else:
            return f"The secret number is {self.even_or_odd}."


# Manage attempts, hints, input validation, and the win or loss conditions
def play_game():
    number_game = NumberGuesser(1, 10)

    attempts_left = 4
    hint_count = 3

    print("=== Number Guessing Game ===")
    print("I'm thinking of a number from 1 to 10.")
    print(f"You have {attempts_left} attempts.")
    print(f"You have {hint_count} hints.")

    while attempts_left > 0:
        player_input = input(
            "\nEnter your guess or type 'hint': "
        )

        # Requesting a hint does not consume a guessing attempt
        if player_input.lower() == "hint":
            if hint_count > 0:
                print(number_game.give_hint())
                hint_count -= 1
            else:
                print("You have already used all of your hints.")

            print(f"Attempts remaining: {attempts_left}")
            print(f"Hints remaining: {hint_count}")
            continue

        # Convert guesses to integers while keeping invalid text from stopping the game
        try:
            player_guess = int(player_input)

        except ValueError:
            print("Please enter a whole number or type 'hint'.")
            continue

        # Reject guesses outside the allowed range without reducing the attempt count
        if (
            player_guess < number_game.minimum
            or player_guess > number_game.maximum
        ):
            print("Please enter a number from 1 to 10.")
            continue

        if player_guess == number_game.secret_number:
            print("Great job! You found the secret number!")
            return

        # Only an incorrect valid guess uses one of the player's attempts
        attempts_left -= 1

        if attempts_left == 0:
            print(
                "Game over! The correct number was "
                f"{number_game.secret_number}."
            )
            return

        print("That's not it! Give it another shot.")
        print(f"Attempts remaining: {attempts_left}")
        print(f"Hints remaining: {hint_count}")


play_game()