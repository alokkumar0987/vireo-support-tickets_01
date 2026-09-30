# Cluster report (k=32, 11,875 tickets, all-MiniLM-L6-v2 + KMeans)

Purity = share of the cluster's tickets whose `theme` equals the cluster's most common theme.

|   cluster |   size | top_terms                                                                             | main_theme                              |   purity | second_theme                            | main_bot_tag        |   bot_tag_share |   repeat_rate |
|----------:|-------:|:--------------------------------------------------------------------------------------|:----------------------------------------|---------:|:----------------------------------------|:--------------------|----------------:|--------------:|
|        29 |    688 | claim, centre, service, service centre, warranty, warranty claim                      | Warranty claim / repair status          |     0.58 | App / firmware update failure           | Warranty & Repair   |            0.47 |          0.33 |
|        26 |    661 | order, delivered, haven, order delivered, received, haven received                    | Delivery delayed / not delivered        |     0.49 | Wrong item / variant received           | Delivery & Shipping |            0.70 |          0.28 |
|        10 |    658 | deducted, deducted order, order, payment deducted, paid, confirmation                 | Double charge / payment failed          |     0.65 | Other                                   | Billing & Payments  |            0.68 |          0.33 |
|        13 |    633 | app, screen, update, device, updated, white                                           | App / firmware update failure           |     0.55 | Screen / touch / hardware fault         | App & Firmware      |            0.54 |          0.32 |
|         5 |    605 | sound, audio, music, mode, wired mode, silent                                         | Audio fault (one side, distortion, mic) |     0.86 | Damaged in transit / dead on arrival    | Audio Quality       |            0.78 |          0.31 |
|        23 |    512 | charging, charge, left, case, charged, does                                           | Battery drain / not charging            |     0.72 | Audio fault (one side, distortion, mic) | Charging & Battery  |            0.77 |          0.31 |
|        14 |    511 | damaged, product, box, photos, strap, took                                            | Damaged in transit / dead on arrival    |     0.67 | Other                                   | Delivery & Shipping |            0.60 |          0.32 |
|        31 |    476 | tracking, delivered tracking, tracking updating, updating, order delivered, delivered | Delivery delayed / not delivered        |     0.96 | Battery drain / not charging            | Delivery & Shipping |            0.82 |          0.27 |
|        21 |    475 | courier, called courier, called, delivered, courier marked, marked delivered          | Delivery delayed / not delivered        |     0.78 | Return pickup not done                  | Delivery & Shipping |            0.61 |          0.32 |
|        18 |    466 | invoice, orders page, checked orders, browser, orders, downloading                    | Invoice / GST                           |     0.98 | Coupon / discount / price               | Billing & Payments  |            0.76 |          0.21 |
|        15 |    447 | phone, pair, laptop, blinks, light, restarted phone                                   | Pairing & connection drops              |     1.00 |                                         | Connectivity        |            0.80 |          0.29 |
|        24 |    428 | refund, refund promised, refund received, waiting refund, emailed, emailed twice      | Refund delayed / not received           |     0.91 | Double charge / payment failed          | Returns & Refunds   |            0.77 |          0.36 |
|        28 |    418 | calls, audio, disconnecting, keeps disconnecting, disconnects, disconnecting minutes  | Audio fault (one side, distortion, mic) |     0.44 | Pairing & connection drops              | Audio Quality       |            0.47 |          0.29 |
|         4 |    418 | list, phone, device, phone just, just doesn, doesn                                    | Pairing & connection drops              |     0.59 | App / firmware update failure           | Connectivity        |            0.78 |          0.32 |
|         6 |    409 | battery, use, fast, drains, battery drains, lasts                                     | Battery drain / not charging            |     0.82 | Double charge / payment failed          | Charging & Battery  |            0.83 |          0.32 |
|         8 |    402 | return, accepted, money hasn, hasn, ago money, hasn come                              | Refund delayed / not received           |     0.74 | Other                                   | Returns & Refunds   |            0.67 |          0.40 |
|         3 |    397 | code, caps, code working, tried caps, checkout, tried                                 | Coupon / discount / price               |     0.82 | Account / OTP login                     | Billing & Payments  |            0.70 |          0.25 |
|        20 |    391 | pickup, rescheduled, pickup twice, rescheduled pickup, happened, pickup happened      | Return pickup not done                  |     0.71 | Delivery delayed / not delivered        | Returns & Refunds   |            0.82 |          0.42 |
|         1 |    384 | shipped, order status, status stuck, stuck shipped, status, stuck                     | Delivery delayed / not delivered        |     0.71 | Address change / wrong address          | Delivery & Shipping |            0.71 |          0.28 |
|         2 |    364 | cancel, cancel button, tried cancel, greyed, button greyed, cancel order              | Order cancellation                      |     0.91 | Wrong item / variant received           | Other               |            0.86 |          0.20 |
|        19 |    338 | vireo, vireo app, app, expected vireo, expected, keeps                                | App / firmware update failure           |     0.34 | Delivery delayed / not delivered        | App & Firmware      |            0.28 |          0.33 |
|        16 |    328 | work, work iphone, iphone, 2019, 2019 samsung, does work                              | Product / compatibility question        |     0.82 | Other                                   | Product Enquiry     |            0.85 |          0.16 |
|         7 |    228 | account, page failed, shows account, failed paid, paid shows, failed                  | Account / OTP login                     |     0.93 | Double charge / payment failed          | Account & Login     |            0.53 |          0.26 |
|        17 |    201 | firmware, firmware update, update, stuck, kept, hours                                 | App / firmware update failure           |     0.95 | Other                                   | App & Firmware      |            0.74 |          0.34 |
|         9 |    182 | address, editing app, editing, tried editing, old flat, going                         | Address change / wrong address          |     0.97 | Delivery delayed / not delivered        | Other               |            0.83 |          0.26 |
|        25 |    143 | otp, otp coming, coming number, coming, tried browser, browser                        | Account / OTP login                     |     0.98 | Battery drain / not charging            | Account & Login     |            0.83 |          0.19 |
|        11 |    143 | fails, keeps cutting, cutting, keeps, time, phone                                     | Pairing & connection drops              |     0.99 | App / firmware update failure           | Connectivity        |            0.83 |          0.33 |
|        12 |    132 | tv, replacement, request, want, refund, order                                         | Product / compatibility question        |     0.98 | Battery drain / not charging            | Product Enquiry     |            0.78 |          0.16 |
|        30 |    129 | rs, says, says orders, site says, orders, went                                        | Other                                   |     0.43 | Double charge / payment failed          | Billing & Payments  |            0.87 |          0.26 |
|        27 |    129 | connect, decoration, decoration point, point, just decoration, just                   | Product / compatibility question        |     0.85 | Audio fault (one side, distortion, mic) | Product Enquiry     |            0.71 |          0.19 |
|        22 |     91 | shows, upi, app shows, app, bank statement, shows tried                               | Double charge / payment failed          |     0.51 | Other                                   | Billing & Payments  |            0.84 |          0.32 |
|         0 |     88 | want, replacement, refund, want replacement, request, order                           | Product / compatibility question        |     0.57 | Other                                   | Product Enquiry     |            0.84 |          0.24 |

## Cluster 0: 88 tickets | want, replacement, refund, want replacement, request, order
- Our theme: **Product / compatibility question** (57%); next: Other | bot tag: Product Enquiry (84%) | repeat rate 24%

## Cluster 1: 384 tickets | shipped, order status, status stuck, stuck shipped, status, stuck
- Our theme: **Delivery delayed / not delivered** (71%); next: Address change / wrong address | bot tag: Delivery & Shipping (71%) | repeat rate 28%

## Cluster 2: 364 tickets | cancel, cancel button, tried cancel, greyed, button greyed, cancel order
- Our theme: **Order cancellation** (91%); next: Wrong item / variant received | bot tag: Other (86%) | repeat rate 20%

## Cluster 3: 397 tickets | code, caps, code working, tried caps, checkout, tried
- Our theme: **Coupon / discount / price** (82%); next: Account / OTP login | bot tag: Billing & Payments (70%) | repeat rate 25%

## Cluster 4: 418 tickets | list, phone, device, phone just, just doesn, doesn
- Our theme: **Pairing & connection drops** (59%); next: App / firmware update failure | bot tag: Connectivity (78%) | repeat rate 32%

## Cluster 5: 605 tickets | sound, audio, music, mode, wired mode, silent
- Our theme: **Audio fault (one side, distortion, mic)** (86%); next: Damaged in transit / dead on arrival | bot tag: Audio Quality (78%) | repeat rate 31%

## Cluster 6: 409 tickets | battery, use, fast, drains, battery drains, lasts
- Our theme: **Battery drain / not charging** (82%); next: Double charge / payment failed | bot tag: Charging & Battery (83%) | repeat rate 32%

## Cluster 7: 228 tickets | account, page failed, shows account, failed paid, paid shows, failed
- Our theme: **Account / OTP login** (93%); next: Double charge / payment failed | bot tag: Account & Login (53%) | repeat rate 26%

## Cluster 8: 402 tickets | return, accepted, money hasn, hasn, ago money, hasn come
- Our theme: **Refund delayed / not received** (74%); next: Other | bot tag: Returns & Refunds (67%) | repeat rate 40%

## Cluster 9: 182 tickets | address, editing app, editing, tried editing, old flat, going
- Our theme: **Address change / wrong address** (97%); next: Delivery delayed / not delivered | bot tag: Other (83%) | repeat rate 26%

## Cluster 10: 658 tickets | deducted, deducted order, order, payment deducted, paid, confirmation
- Our theme: **Double charge / payment failed** (65%); next: Other | bot tag: Billing & Payments (68%) | repeat rate 33%

## Cluster 11: 143 tickets | fails, keeps cutting, cutting, keeps, time, phone
- Our theme: **Pairing & connection drops** (99%); next: App / firmware update failure | bot tag: Connectivity (83%) | repeat rate 33%

## Cluster 12: 132 tickets | tv, replacement, request, want, refund, order
- Our theme: **Product / compatibility question** (98%); next: Battery drain / not charging | bot tag: Product Enquiry (78%) | repeat rate 16%

## Cluster 13: 633 tickets | app, screen, update, device, updated, white
- Our theme: **App / firmware update failure** (55%); next: Screen / touch / hardware fault | bot tag: App & Firmware (54%) | repeat rate 32%

## Cluster 14: 511 tickets | damaged, product, box, photos, strap, took
- Our theme: **Damaged in transit / dead on arrival** (67%); next: Other | bot tag: Delivery & Shipping (60%) | repeat rate 32%

## Cluster 15: 447 tickets | phone, pair, laptop, blinks, light, restarted phone
- Our theme: **Pairing & connection drops** (100%); next: - | bot tag: Connectivity (80%) | repeat rate 29%

## Cluster 16: 328 tickets | work, work iphone, iphone, 2019, 2019 samsung, does work
- Our theme: **Product / compatibility question** (82%); next: Other | bot tag: Product Enquiry (85%) | repeat rate 16%

## Cluster 17: 201 tickets | firmware, firmware update, update, stuck, kept, hours
- Our theme: **App / firmware update failure** (95%); next: Other | bot tag: App & Firmware (74%) | repeat rate 34%

## Cluster 18: 466 tickets | invoice, orders page, checked orders, browser, orders, downloading
- Our theme: **Invoice / GST** (98%); next: Coupon / discount / price | bot tag: Billing & Payments (76%) | repeat rate 21%

## Cluster 19: 338 tickets | vireo, vireo app, app, expected vireo, expected, keeps
- Our theme: **App / firmware update failure** (34%); next: Delivery delayed / not delivered | bot tag: App & Firmware (28%) | repeat rate 33%

## Cluster 20: 391 tickets | pickup, rescheduled, pickup twice, rescheduled pickup, happened, pickup happened
- Our theme: **Return pickup not done** (71%); next: Delivery delayed / not delivered | bot tag: Returns & Refunds (82%) | repeat rate 42%

## Cluster 21: 475 tickets | courier, called courier, called, delivered, courier marked, marked delivered
- Our theme: **Delivery delayed / not delivered** (78%); next: Return pickup not done | bot tag: Delivery & Shipping (61%) | repeat rate 32%

## Cluster 22: 91 tickets | shows, upi, app shows, app, bank statement, shows tried
- Our theme: **Double charge / payment failed** (51%); next: Other | bot tag: Billing & Payments (84%) | repeat rate 32%

## Cluster 23: 512 tickets | charging, charge, left, case, charged, does
- Our theme: **Battery drain / not charging** (72%); next: Audio fault (one side, distortion, mic) | bot tag: Charging & Battery (77%) | repeat rate 31%

## Cluster 24: 428 tickets | refund, refund promised, refund received, waiting refund, emailed, emailed twice
- Our theme: **Refund delayed / not received** (91%); next: Double charge / payment failed | bot tag: Returns & Refunds (77%) | repeat rate 36%

## Cluster 25: 143 tickets | otp, otp coming, coming number, coming, tried browser, browser
- Our theme: **Account / OTP login** (98%); next: Battery drain / not charging | bot tag: Account & Login (83%) | repeat rate 19%

## Cluster 26: 661 tickets | order, delivered, haven, order delivered, received, haven received
- Our theme: **Delivery delayed / not delivered** (49%); next: Wrong item / variant received | bot tag: Delivery & Shipping (70%) | repeat rate 28%

## Cluster 27: 129 tickets | connect, decoration, decoration point, point, just decoration, just
- Our theme: **Product / compatibility question** (85%); next: Audio fault (one side, distortion, mic) | bot tag: Product Enquiry (71%) | repeat rate 19%

## Cluster 28: 418 tickets | calls, audio, disconnecting, keeps disconnecting, disconnects, disconnecting minutes
- Our theme: **Audio fault (one side, distortion, mic)** (44%); next: Pairing & connection drops | bot tag: Audio Quality (47%) | repeat rate 29%

## Cluster 29: 688 tickets | claim, centre, service, service centre, warranty, warranty claim
- Our theme: **Warranty claim / repair status** (58%); next: App / firmware update failure | bot tag: Warranty & Repair (47%) | repeat rate 33%

## Cluster 30: 129 tickets | rs, says, says orders, site says, orders, went
- Our theme: **Other** (43%); next: Double charge / payment failed | bot tag: Billing & Payments (87%) | repeat rate 26%

## Cluster 31: 476 tickets | tracking, delivered tracking, tracking updating, updating, order delivered, delivered
- Our theme: **Delivery delayed / not delivered** (96%); next: Battery drain / not charging | bot tag: Delivery & Shipping (82%) | repeat rate 27%