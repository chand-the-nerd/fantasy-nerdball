"""Token management utility for FPL chips and game state."""


class TokenManager:
    """Handles FPL token validation, adjustments, and logic."""
    
    def __init__(self, config):
        """Initialize with config validation and adjustments."""
        self.config = config
        self._validate_tokens()
        self._adjust_for_tokens()
    
    def _validate_tokens(self):
        """Validate that only one chip token is active at a time."""
        active_chips = [
            self.config.WILDCARD,
            self.config.BENCH_BOOST, 
            self.config.TRIPLE_CAPTAIN
        ]
        
        active_count = sum(active_chips)
        
        if active_count > 1:
            chip_names = []
            if self.config.WILDCARD:
                chip_names.append("WILDCARD")
            if self.config.BENCH_BOOST:
                chip_names.append("BENCH_BOOST")
            if self.config.TRIPLE_CAPTAIN:
                chip_names.append("TRIPLE_CAPTAIN")
            
            raise ValueError(
                f"❌ ERROR: Multiple chips active: {', '.join(chip_names)}. "
                f"Only one chip can be used per gameweek."
            )
        
        if active_count == 1:
            active_chip = None
            if self.config.WILDCARD:
                active_chip = "WILDCARD"
            elif self.config.BENCH_BOOST:
                active_chip = "BENCH_BOOST"  
            elif self.config.TRIPLE_CAPTAIN:
                active_chip = "TRIPLE_CAPTAIN"
            
            print(f"🃏 CHIP ACTIVE: {active_chip}")
    
    def _adjust_for_tokens(self):
        """Adjust settings based on active tokens."""
        if self.config.WILDCARD:
            # Wildcard gives unlimited transfers
            self.config.FREE_TRANSFERS = 15
            print(f"🃏 WILDCARD: Free transfers set to "
                  f"{self.config.FREE_TRANSFERS}")
        
        if self.config.FREE_HIT_PREV_GW:
            print("🔄 FREE HIT USED PREVIOUS GW: Squad will be loaded from "
                  f"GW{self.config.GAMEWEEK - 2}")
        
        if self.config.BENCH_BOOST:
            print("💪 BENCH BOOST: All 15 players will count towards points")
        
        if self.config.TRIPLE_CAPTAIN:
            print("👑 TRIPLE CAPTAIN: Captain will score 3x points")
    
    def get_previous_squad_gameweek(self):
        """
        Get the gameweek to load previous squad from.
        
        Returns:
            int: Gameweek number to load squad from
        """
        if self.config.FREE_HIT_PREV_GW:
            return self.config.GAMEWEEK - 2
        else:
            return self.config.GAMEWEEK - 1
    
    def calculate_captain_multiplier(self):
        """
        Get the captain points multiplier based on active chips.
        
        Returns:
            int: Captain multiplier (2 for normal, 3 for Triple Captain)
        """
        return 3 if self.config.TRIPLE_CAPTAIN else 2
    
    def get_captain_display_text(self):
        """
        Get the captain display text based on active chips.
        
        Returns:
            str: Display text for captain multiplier
        """
        return "x3" if self.config.TRIPLE_CAPTAIN else "x2"
    
    def should_include_bench_points(self):
        """
        Check if bench points should be included in total.
        
        Returns:
            bool: True if Bench Boost is active
        """
        return self.config.BENCH_BOOST
    
    def get_points_label(self):
        """
        Get appropriate points label based on active chips.
        
        Returns:
            str: Points label for display
        """
        if self.config.BENCH_BOOST:
            return "Full Squad Points (Bench Boost)"
        elif self.config.TRIPLE_CAPTAIN:
            return "Starting XI Points (Triple Captain)"
        else:
            return "Starting XI Points"