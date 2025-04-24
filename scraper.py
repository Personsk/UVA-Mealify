import re
import csv
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


def extract_calories(calories_str):
    """Extract numeric calorie value from string like '190 Calories'"""
    match = re.search(r'(\d+)', calories_str)
    if match:
        return int(match.group(1))
    return 0


def get_menu_items(url, dining_hall_name):
    """Get menu items from a specific dining hall URL"""
    # Setup headless Chrome
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    try:
        # Navigate to the page
        driver.get(url)
        print(f"Fetching menu items from {dining_hall_name}...")

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

                # Skip items with 0 calories
                if calories <= 0:
                    continue

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
                    "description": description,
                    "dining_hall": dining_hall_name
                })
            except Exception as e:
                print(f"Error processing an item: {e}")

        return results

    finally:
        # Clean up
        driver.quit()


def save_to_csv(menu_items, filename="uva_dining_menu.csv"):
    """Save menu items to CSV, avoiding redundancy based on item name"""
    # Check if file exists and load existing items to avoid redundancy
    existing_items = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Use name as the key to check for redundancy
                    existing_items[row['name']] = row
        except Exception as e:
            print(f"Error reading existing CSV: {e}")

    # Prepare to write to CSV
    fieldnames = ['name', 'calories', 'dining_hall', 'description']

    # Determine which items to add (non-redundant)
    new_items = []
    updated_items = []
    for item in menu_items:
        if item['name'] not in existing_items:
            new_items.append(item)
        else:
            # If the item exists but from a different dining hall, update to add the new location
            existing_item = existing_items[item['name']]
            if existing_item['dining_hall'] != item['dining_hall']:
                existing_item['dining_hall'] = f"{existing_item['dining_hall']}, {item['dining_hall']}"
                updated_items.append(existing_item)

    # If there are no new or updated items, inform the user
    if not new_items and not updated_items:
        print("No new menu items to add.")
        return 0

    # Combine existing (excluding updated ones), new, and updated items
    items_to_write = []
    for name, item in existing_items.items():
        # Only include if not in updated_items
        if name not in [updated['name'] for updated in updated_items]:
            items_to_write.append(item)

    items_to_write.extend(new_items)
    items_to_write.extend(updated_items)

    # Write all items to CSV
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in items_to_write:
                # Only write the fields we care about
                writer.writerow({
                    'name': item['name'],
                    'calories': item['calories'],
                    'dining_hall': item['dining_hall'],
                    'description': item['description']
                })

        return len(new_items)
    except Exception as e:
        print(f"Error writing to CSV: {e}")
        return 0


def main():
    # Define dining hall URLs
    dining_halls = {
        "Runk": "https://virginia.campusdish.com/LocationsAndMenus/Runk",
        "O-Hill": "https://virginia.campusdish.com/LocationsAndMenus/ObservatoryHillDiningRoom",
        "Newcomb": "https://virginia.campusdish.com/en/locationsandmenus/freshfoodcompany/"
    }

    # Fetch menu items from all dining halls
    all_menu_items = []

    for hall_name, url in dining_halls.items():
        try:
            hall_items = get_menu_items(url, hall_name)
            all_menu_items.extend(hall_items)
            print(f"Found {len(hall_items)} menu items with calories from {hall_name}.")
        except Exception as e:
            print(f"Error fetching items from {hall_name}: {e}")

    if not all_menu_items:
        print("No menu items found. Please check the websites or try again later.")
        return

    print(f"Found a total of {len(all_menu_items)} menu items with calories across all dining halls.")

    # Ask for filename
    default_filename = "uva_dining_menu.csv"
    filename = input(f"\nEnter filename for CSV (default: {default_filename}): ").strip()
    if not filename:
        filename = default_filename
    if not filename.endswith('.csv'):
        filename += '.csv'

    # Save to CSV
    print(f"Saving menu items to {filename}...")
    new_items_count = save_to_csv(all_menu_items, filename)

    print(f"CSV file updated. Added {new_items_count} new menu items.")
    print(f"Data saved to {os.path.abspath(filename)}")


if __name__ == "__main__":
    main()
