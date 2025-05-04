
#Decorator Pattern
#Purpose: Adds new responsibilities to objects dynamically without altering their structure.
#Use case: UI toolkits, middleware frameworks, stream handling (like Java I/O).
#Benefit: Supports open/closed principle (open for extension, closed for modification).

class EmailNotifier:
    def send(self):
        print("Email notification")

class SmsNotifier(EmailNotifier):
    def __init__(self, emailNotifier:EmailNotifier):
        self.emailNotifier = emailNotifier
    def send(self):
        self.emailNotifier.send()
        print("Email notification")

class TeamsNotifier(SmsNotifier):
    def __init__(self, emailNotifier: EmailNotifier, smsNotifier: SmsNotifier):
        super().__init__(emailNotifier)
        self.smsNotifier = smsNotifier
    def send(self):
        self.smsNotifier.send()
        print("Teams notification")


teamsNotifier = TeamsNotifier(EmailNotifier(), SmsNotifier())
teamsNotifier.send()
