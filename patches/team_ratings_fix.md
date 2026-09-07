# Fix: `ZeroDivisionError` in `calculate_team_ratings.py`

## The bug

In `calculate_team_defensive_rating`, around line 218:

```python
defensive_rating = total_expected_conceded / total_actual_conceded

# Handle edge case of zero goals conceded
if total_actual_conceded == 0:
    defensive_rating = 2.0
```

The guard is *after* the division it is meant to guard. Any team yet to concede
divides by zero and the whole run dies — which is why this breaks early in a
season, when clean sheets are common.

The check above it only tests `total_expected_conceded`, the numerator, which
is never the problem.

## The fix

Move the check before the division:

```python
        if fixture_count == 0 or total_expected_conceded == 0:
            return 1.0

        # A side yet to concede is rated at the ceiling: there is no ratio to
        # take, and conceding nothing is as good as a defence gets.
        if total_actual_conceded == 0:
            return 2.0

        # Higher is better: conceding less than expected.
        defensive_rating = total_expected_conceded / total_actual_conceded

        # Apply reasonable bounds (0.5 to 2.0)
        defensive_rating = max(0.5, min(2.0, defensive_rating))
```

Both early returns are already inside the 0.5–2.0 bounds, so nothing downstream
changes.

## The attacking rating is fine

I checked, expecting the same bug. It isn't there: that function divides by
`total_expected_goals`, which the guard above it already tests. Only the
defensive rating divides by a quantity nothing has checked.
