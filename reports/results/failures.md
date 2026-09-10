# Failure analysis (system: `agent`)

## 1. Intent confusions (51 of 220 cases wrong)

### `flight_disruption` -> `service_complaint`  (5 cases, 10% of all errors)

- why did your member of staff tell us flight was cancelled because of bad weather at Gatwick but easyJet are flying all day?
- Shocking service from and that's consecutive delays using them. Ryan Air in suits!
- I’m interested in how you indend “improving” the flight delay. Shouldn’t you be working on “reducing” it instead? <url>

### `service_complaint` -> `praise_or_chatter`  (4 cases, 8% of all errors)

- thank you for the disappointing experience on our flight #poor #neveragain #ripoff #disappointed
- let's hope. Having had to pay for a seat on long haul in January which is wrong, I hope you manage to bring back the glory. A great airline 🛫🛬 <url>
- That time misconfigured the business seating & curtains wherein you lost 30min to them realising & rectifying #airlols <url>

### `service_complaint` -> `compensation_refund`  (3 cases, 6% of all errors)

- Have any update? Case reference: 17095662
- Don't ever fly with them. They are unprofessional and lost our flights then up charged us to just get a flight back home
- Four weeks on and still no reply from BA customer services about poor practice & standards in Warsaw

### `digital_technical` -> `booking_change_cancel`  (3 cases, 6% of all errors)

- hi there I have tried to book a Paris flight w/ the polo mailing received 25%off but it does not seem to work :/
- cannot change a flight I keep getting an error message! can someone contact me have been on hold for over 1 hour!
- I changed some flights in the app yesterday evening and it’s still showing my original flight times etc despite have received a confirmation email. When does it update in the app?

### `other` -> `booking_change_cancel`  (2 cases, 4% of all errors)

- it will be my BFs first ever flight with BA today. LHR-HKG BA027 Seat 28D. I hope there is something you can do to make it extra special? 🙏 <url>
- Hi, please can you help me identify my BA flight number on a recent codeshare flight with Qatar Airways?

### `checkin_boarding` -> `baggage_policy`  (2 cases, 4% of all errors)

- how early can one check luggage at LHR? If My flight is at 3:15 PM, can I check luggage at LHR at 8:00 am?
- how early can I check my bag through for a 4pm flight at Glasgow?

### `service_complaint` -> `booking_change_cancel`  (2 cases, 4% of all errors)

- definitely avoiding I paid more money to adjust my flight one way then I paid to start with it for the whole journey !!
- Wow, 47 calls ahead of me with . I just received an emailed ticket from with no explanation.

### `praise_or_chatter` -> `service_complaint`  (2 cases, 4% of all errors)

- excellent service on board 1435 this morning despite long delays on the ground. Crew did all they could to keep us informed
- After an impromptu visit to hospital in France, can't thank enough for helping change our flights and getting us home.

## 2. Triage errors

- **False auto-handle (costly): 14** -- needed a human, the bot answered.
- False escalate (cheap): 25 -- answerable, but sent to a human.

### Every false auto-handle

- **hi. I can’t check in on the app or website, it keeps looping back to advance passenger information. Is it just me?**
  - agent said: _App check-in issue, can be answered with public policy from evidence._
  - drafted: _We're sorry you're having issues with check in. If it's still not working, you'll have to check in at the airport._
- **Hi BA I can't seem to download my boarding pass to my phone. Can you help me out please?**
  - agent said: _Boarding pass download issue, handled via public troubleshooting policy._
  - drafted: _Have you got the latest version of the app? If problems persist, the airport will be able to print your boarding pass._
- **Yet another disaster with , flight got delayed by 24 hours**
  - agent said: _Delay notification/complaint without money asked._
  - drafted: _We're sorry about the delay. The local teams will advise you and rebook if necessary._
- **why wont the app let me check in 😩 worked for the rest of the group! #getitsorted <url>**
  - agent said: __
  - drafted: _We're sorry you're having problems checking in online. Our Airport team will be best placed to help you with this. Make sure you arrive early and our staff will be happy to assist._
- **argh can't check in online for Dubai Heathrow club world flight in 8 hrs or pick seats disappointing!**
  - agent said: __
  - drafted: _We're sorry you're unable to check in online. If it's still not working, you'll need to check-in at the airport. We're sorry for the inconvenience._
- **I left my iPad in the seat pocket of my flight yesterday. How do I get it back please?**
  - agent said: __
  - drafted: _I’m afraid you will need to contact the airport where your flight landed, as all items found on board will be handed in there. You can find contact details on our website._
- **trying to check in on app. Not working & says “still waiting for authorisation from the US gov”. How can I check in?**
  - agent said: _Customer is experiencing an online check-in error and seeking assistance._
  - drafted: _We're sorry you're having issues with check-in. If online check-in or the app is not working, you'll need to check in at the airport._
- **UK agent here. Is there a system issue with pre-booking seats with AA codeshare flights? Trying to avoid calling Rapport!**
  - agent said: _Question about codeshare flights from a trade agent._
  - drafted: _Hi there. If you are experiencing persistent issues online, you can check our website or speak directly with our sales team._
- **Hi BA, how do I get a receipt for a seat selection I made to a current booking? There is no option in Manage My Booking?**
  - agent said: __
  - drafted: _If you've not received your receipt automatically you'll need to request one online using our website._
- **can’t get my boarding pass on my phone and don’t have a printer, what shall I do?**
  - agent said: __
  - drafted: _We're sorry for the difficulties. If you cannot get your boarding pass on your phone and do not have a printer, the airport can print your boarding pass._
- **. I booked my flights with . They said I need to reconfirm my booking with you. How can I do this? Many thanks!**
  - agent said: __
  - drafted: _You can confirm your flights by logging into Manage My Booking on our website._
- **I am flying to NYC from Heathrow next week and was just looking at meal option as my missus is gluten free and vegetarian, but it doesn’t give us an option for this. #fussyeater #cantwait #birthday**
  - agent said: __
  - drafted: _Hi, we're sorry for any difficulty finding meal options. You can check our website for details on special dietary requirements._
- **- what's going on with your mobile app? Can't get my boarding pass downloaded for tomorrow - plus, every time you change a seat, you have to re-enter your mobile number and email!!!! Twice a week!**
  - agent said: __
  - drafted: _Have you got the latest version of the app? If problems persist, the airport will be able to print your boarding pass._
- **Minor name change on a domestic flight (replacing I and EA), does this need corrected and can this be done at airport?**
  - agent said: __
  - drafted: _Yes, name corrections need to be done via the phone. This can't be done online._

## 3. Lowest-scoring drafted replies

### mean 1.00  (grou=1 help=1 tone=1 safe=1)

- customer: your agents made people check bags because the flight is full, yet many people have taken up overhead bins with coats.
- agent:    Your allowance depends on your ticket type and cabin. We're sorry for any frustration with baggage checks.
- historical: Hi Leigh, we do this if there's a chance there won't be enough space for everyone. It's much harder to fix when you're all on-board! 1/2
- judge critique: __

### mean 1.00  (grou=1 help=1 tone=1 safe=1)

- customer: I do love my card so much that I am taking it on a holiday with my favourite carrier #fintech at its best. <url>
- agent:    We're pleased to see you're making full use of our facilities and wish you a pleasant flight.
- historical: That’s really nice to hear, Pedro. Where are we taking you this time? ^Monica
- judge critique: __

### mean 1.00  (grou=1 help=1 tone=1 safe=1)

- customer: Who do i speak to about compensation for a 4.5 hour delayed flight from Rome and a damaged suitcase?
- agent:    Thanks for getting in touch. If you can send us more details by DM, we'll take a look for you.
- historical: Sorry to hear about your experience when travelling with us, Em. Please use this link: <url> ^Gill
- judge critique: __

### mean 1.00  (grou=1 help=1 tone=1 safe=1)

- customer: First time flying today GLA > LCY - very impressed, excellent service and complimentary refreshments, unexpected bonus! ✈️😀
- agent:    We're pleased to hear you had a great first flight with us! We hope you have a nice evening.
- historical: Hi Ian. We're glad you enjoyed your first time flying with us. Little surprises always make it that little bit more fun. ^N
- judge critique: __

### mean 1.00  (grou=1 help=1 tone=1 safe=1)

- customer: Hello, thinking of booking a flight from Barcelona to Heathrow for tomorrow morning. Will this still be scheduled to fly?
- agent:    You can check the live status of your flight on our website before you travel.
- historical: Yes Emma. All flights are showing as operating on time on our website. ^Gill
- judge critique: __

### mean 1.00  (grou=1 help=1 tone=1 safe=1)

- customer: Thanks for the upgrade at the end of a rubbish day. I return on the 13th if you want to complete the effect 😜 <url>
- agent:    That's our pleasure! We hope you enjoyed your time on board travelling with us.
- historical: The world has a way of making things right... who knows! I've got my fingers crossed for you. ^Kimbers
- judge critique: __

## 4. Groundedness by judge dimension (agent)

- `groundedness` score distribution -> 1:11  2:4  3:3  4:18  5:24
- `helpfulness` score distribution -> 1:14  2:15  3:9  4:18  5:4
- `tone` score distribution -> 1:10  2:0  3:10  4:33  5:7
- `safety` score distribution -> 1:13  2:0  3:0  4:0  5:47

