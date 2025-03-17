import random

training_data = [
    (1200, 100, 100, 2.8),
    (1500, 100, 100, 3.5),
    (900, 80, 100, 2.0),
    (1100, 80, 100, 2.5),
    (800, 60, 30, 1.4),
    (1000, 60, 30, 1.8),
    (600, 50, 25, 1.0),
    (750, 50, 25, 1.2),
    (1300, 100, 100, 3.1),
    (1000, 80, 100, 2.2),
    (900, 60, 30, 1.6),
    (700, 50, 25, 1.1),
    (1400, 100, 100, 3.3),
    (1200, 80, 100, 2.7),
    (1100, 60, 30, 1.9),
    (800, 50, 25, 1.3)
]

# Generate 50 new data points
for _ in range(50):
    size = random.randint(500, 2000)  # Size in sq feet
    loc_value = random.choice([100, 80, 60, 50])  # Location value
    if loc_value == 100 or loc_value == 80:
        prox = 100
    else:
        prox = loc_value / 2
    actual_price = random.uniform(1.0, 4.0)  # Actual price
    training_data.append((size, loc_value, prox, actual_price))

print(f"Added 50 data points. Total data points: {len(training_data)}")
# Now training_data has 66 data points (16 original + 50 new)

learning_rate = 0.0000001
epochs = 1000
decay_rate = 0.99

def polynomial_regression(training_data):
    polynomial_data = []
    for size, loc_value, prox, actual_price in training_data:
        polynomial_data.append((size, loc_value, prox, size**2, loc_value**2, size*loc_value, actual_price))
    return polynomial_data
training_data_poly = polynomial_regression(training_data)


actual_prices = tuple(price for _, _, _, price in training_data)

loc_input = input("Enter location (Metropolis, City, Town, Village): ").strip().capitalize()

if loc_input == "Metropolis":
    loc_value = 100
elif loc_input == "City":
    loc_value = 80
elif loc_input == "Town":
    loc_value = 60
elif loc_input == "Village":
    loc_value = 50
else:
    print("Invalid location entered.")
    exit()

size = int(input("Enter size in sq feet: "))

if loc_value == 100 or loc_value == 80:
    prox = 100
else:
    prox = loc_value / 2

w1 = 1
w2 = 2
w3 = 1.5
w4 = 1  # size_squared
w5 = 2  # loc_squared
w6 = 1.5 # size_loc

def landPrice(size, loc_value, prox, size_squared, loc_squared, size_loc):
    price = (size * w1 + prox * w2 + loc_value * w3 + size_squared * w4 + loc_squared * w5 + size_loc * w6)/1000
    return price

predicted_prices = []
for size, loc_value, prox, size_squared, loc_squared, size_loc,  _ in training_data_poly:
    predicted_price = landPrice(size, loc_value, prox, size_squared, loc_squared, size_loc)
    predicted_prices.append(predicted_price)

print("Predicted Prices:", predicted_prices)
print(f"Actual prices:{actual_prices}")

def calculate_cost(predicted_prices, actual_prices):
    if len(predicted_prices) != len(actual_prices):
        raise ValueError("Predicted and actual price lists must have the same length.")

    squared_errors = [(predicted - actual) ** 2 for predicted, actual in zip(predicted_prices, actual_prices)]
    mean_squared_error = sum(squared_errors) / len(squared_errors)
    return mean_squared_error

cost = calculate_cost(predicted_prices, actual_prices)
print("Mean Squared Error (MSE):", cost)

def calculate_gradients(predicted_prices, actual_prices, training_data):
    dw1 = 0
    dw2 = 0
    dw3 = 0
    dw4 = 0
    dw5 = 0
    dw6 = 0
    n = len(training_data)

    for i in range(n):
        size, loc_value, prox, size_squared, loc_squared, size_loc, _ = training_data[i]
        error = predicted_prices[i] - actual_prices[i]
        dw1 = dw1 + 2 * error * size
        dw2 = dw2 + 2 * error * prox
        dw3 = dw3 + 2 * error * loc_value
        dw4 = dw4 + 2 * error * size_squared
        dw5 = dw5 + 2 * error * loc_squared
        dw6 = dw6 + 2 * error * size_loc

    dw1 = dw1 / n
    dw2 = dw2 / n
    dw3 = dw3 / n
    dw4 = dw4 / n
    dw5 = dw5 / n 
    dw6 = dw6 / n

    return dw1, dw2, dw3, dw4, dw5, dw6

def update_weights(w1, w2, w3, w4, w5, w6, dw1, dw2, dw3, dw4, dw5, dw6,  learning_rate):
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    w3 = w3 - learning_rate * dw3
    w4 = w4 - learning_rate * dw4
    w5 = w5 - learning_rate * dw5
    w6 = w6 - learning_rate * dw6
    return w1, w2, w3, w4, w5, w6

for epoch in range(epochs):
    predicted_prices = [] # Reset the lists at the start of each epoch
    actual_prices = []

    for size, loc_value, prox, size_squared, loc_squared, size_loc, actual_price in training_data_poly:
        predicted_price = landPrice(size, loc_value, prox, size_squared, loc_squared, size_loc)
        predicted_prices.append(predicted_price)
        actual_prices.append(actual_price)

    dw1, dw2, dw3, dw4, dw5, dw6 = calculate_gradients(predicted_prices, actual_prices, training_data_poly)
    w1, w2, w3, w4, w5, w6 = update_weights(w1, w2, w3, w4, w5, w6, dw1, dw2, dw3, dw4, dw5, dw6, learning_rate)

    learning_rate *= decay_rate

    if epoch % 100 == 0:
        
        cost = calculate_cost(predicted_prices, actual_prices)
        print(f"Epoch :{epoch} , Cost :{cost}")

print(f"w1 :{w1}, w2 :{w2}, w3 :{w3}, w4: {w4}, w5: {w5}, w6: {w6}")




