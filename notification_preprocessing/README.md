# Notification preprocessing

This code orders reservation notifications over time and compares each update with the previous state. It converts changes and cancellations into room increases or decreases by lead time, with optional aggregation by check-in date and lead time.

Input CSV files use general reservation, notification, timestamp, stay-date, and room-count columns. Cancellation rows use the generic notification type `cancel`.
