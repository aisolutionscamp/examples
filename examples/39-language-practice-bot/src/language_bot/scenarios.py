from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    key: str
    setting: str
    bot_role: str
    learner_role: str


SCENARIOS = {
    s.key: s
    for s in [
        Scenario("restaurant", "a small restaurant at lunchtime", "waiter", "guest ordering food"),
        Scenario("train", "the ticket desk of a train station", "ticket agent", "traveller buying a ticket"),
        Scenario("doctor", "a doctor's reception desk", "receptionist", "patient making an appointment"),
        Scenario("smalltalk", "a coffee break with a new colleague", "colleague", "new colleague"),
        Scenario("hotel", "a hotel reception in the evening", "receptionist", "guest checking in"),
    ]
}
