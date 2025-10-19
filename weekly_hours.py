"""
Simple script to track daily work hours and calculate weekly total.
Fill in the hours_list with your hours for each day of the week.
Use format "H:MM" (e.g., "8:17" for 8 hours 17 minutes) or just hours as decimal.
"""

def time_to_hours(time_str):
    """
    Convert time string to decimal hours.
    Accepts "H:MM" format or decimal number.
    """
    if isinstance(time_str, (int, float)):
        return float(time_str)
    
    if ':' in str(time_str):
        parts = str(time_str).split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        return hours + minutes / 60.0
    else:
        return float(time_str)


def calculate_weekly_hours(hours_list):
    """
    Calculates total hours worked from a list of daily hours.
    
    Args:
        hours_list: List of 7 values representing hours worked 
                    (Monday through Sunday)
                    Can be "H:MM" strings or decimal numbers
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    if len(hours_list) != 7:
        print(f"Error: Please provide exactly 7 values (got {len(hours_list)})")
        return
    
    # Convert all times to decimal hours
    decimal_hours = [time_to_hours(h) for h in hours_list]
    
    # Calculate total
    total_hours = sum(decimal_hours)
    total_hours_int = int(total_hours)
    total_minutes = round((total_hours - total_hours_int) * 60)
    
    # Display results
    print("\n" + "="*40)
    print("WEEKLY HOURS SUMMARY")
    print("="*40)
    for day, hours in zip(days, decimal_hours):
        hours_int = int(hours)
        minutes = round((hours - hours_int) * 60)
        print(f"{day:12s}: {hours_int:2d}h {minutes:02d}m ({hours:.2f} hours)")
    print("-"*40)
    print(f"{'TOTAL':12s}: {total_hours_int:2d}h {total_minutes:02d}m ({total_hours:.2f} hours)")
    print("="*40)
    
    return total_hours


if __name__ == "__main__":
    # Fill in your hours here (Monday through Sunday)
    # Use "H:MM" format (e.g., "8:17") or decimal hours (e.g., 8.5)
    hours_list = ["8:17", "8:42", "8:52", "3:40", "5:30", 0, 0]
    
    calculate_weekly_hours(hours_list)
