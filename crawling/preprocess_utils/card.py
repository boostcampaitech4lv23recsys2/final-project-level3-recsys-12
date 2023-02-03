import pandas as pd


class Card:
    def __init__(self, card, house_card):
        self.card = card
        self.house_card = house_card

    def preprocessing(self):
        hc_temp = {
            i: j
            for i, j in zip(self.house_card.card.values, self.house_card.house.values)
        }
        self.card["house_id"] = self.card.card.map(hc_temp)

        return self.card
