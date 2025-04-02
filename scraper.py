import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from itertools import combinations



def extract_calories(calories_str):
    """Extract numeric calorie value from string like '190 Calories'"""
    match = re.search(r'(\d+)', calories_str)
    if match:
        return int(match.group(1))
    return 0


def get_menu_items():
    # Setup headless Chrome
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    try:
        # Navigate to the page
        url = "https://virginia.campusdish.com/LocationsAndMenus/Runk"
        driver.get(url)

        # Wait for React to render the menu items
        wait = WebDriverWait(driver, 10)
        menu_items = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-testid="product-card"]')))

        # Extract menu data
        results = []
        for item in menu_items:
            try:
                name = item.find_element(By.CSS_SELECTOR, '[data-testid="product-card-header-link"]').text
                calories_element = item.find_element(By.CSS_SELECTOR,
                                                     '[data-testid="product-card-info-block-calories"]')
                calories_text = calories_element.text if calories_element else "0 Calories"
                calories = extract_calories(calories_text)

                # Try to get description - might be collapsed/hidden
                try:
                    description_element = item.find_element(By.CSS_SELECTOR, '[data-testid="product-card-description"]')
                    description = description_element.text
                except:
                    description = "No description available"

                results.append({
                    "name": name,
                    "calories": calories,
                    "calories_text": calories_text,
                    "description": description
                })
            except Exception as e:
                print(f"Error processing an item: {e}")

        return results

    finally:
        # Clean up
        driver.quit()


def create_meal_plans(menu_items, calorie_limit):
    """Create meal combinations that fall under the calorie limit"""
    # Filter out items with missing or zero calories
    valid_items = [item for item in menu_items if item['calories'] > 0]

    # Sort items by calories (ascending) to make combinations more efficient
    valid_items.sort(key=lambda x: x['calories'])

    meal_plans = []

    # Try combinations of 1-3 items
    for num_items in range(1, 4):
        for combo in combinations(valid_items, num_items):
            total_calories = sum(item['calories'] for item in combo)
            if total_calories <= calorie_limit:
                meal_plans.append({
                    'items': combo,
                    'total_calories': total_calories
                })

    # Sort meal plans by total calories (descending) to get the most filling options first
    meal_plans.sort(key=lambda x: x['total_calories'], reverse=True)

    # Return top 10 meal plans
    return meal_plans[:10]


def main():
    # Fetch menu items
    print("Fetching menu items from the dining website...")
    menu_items = get_menu_items()

    if not menu_items:
        print("No menu items found. Please check the website or try again later.")
        return

    print(f"Found {len(menu_items)} menu items.")

    # Ask for calorie limit
    while True:
        try:
            calorie_limit = int(input("\nEnter your calorie limit for a meal: "))
            if calorie_limit <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    # Create meal plans
    print(f"\nCreating meal plans under {calorie_limit} calories...")
    meal_plans = create_meal_plans(menu_items, calorie_limit)

    if not meal_plans:
        print(f"No meal combinations found under {calorie_limit} calories.")
        return

    # Display meal plans
    print(f"\nTop {len(meal_plans)} meal options under {calorie_limit} calories:\n")

    for i, plan in enumerate(meal_plans, 1):
        print(f"Meal Option {i} - Total: {plan['total_calories']} calories")
        for item in plan['items']:
            print(f"  • {item['name']} ({item['calories_text']})")
            if item['description'] != "No description available":
                print(f"    {item['description']}")
        print()


if __name__ == "__main__":
    main()