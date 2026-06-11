## strata_fp8_vs_bf16_c2

baseline `claude_bf16_vosplit_c2_ctx8191` vs comparator `claude_fp8_c2_ctx8191` — 8190 tokens, mean delta **-0.1643** nats (negative = quantized better)

| baseline-NLL band | count | mean delta | share of total |
|---|---|---|---|
| [0,0.5) | 2426 | +0.0285 | -5.1% |
| [0.5,1) | 316 | +0.1565 | -3.7% |
| [1,2) | 422 | -0.0411 | +1.3% |
| [2,4) | 745 | -0.1544 | +8.5% |
| [4,8) | 1474 | -0.2048 | +22.4% |
| >=8 | 2807 | -0.3669 | +76.5% |

| positions | count | mean delta | share of total |
|---|---|---|---|
| 0-1022 | 1022 | +0.0333 | -2.5% |
| 1023-2045 | 1023 | -0.1274 | +9.7% |
| 2046-3068 | 1023 | -0.1314 | +10.0% |
| 3069-4091 | 1023 | -0.2022 | +15.4% |
| 4092-5114 | 1023 | -0.1647 | +12.5% |
| 5115-6137 | 1023 | -0.2084 | +15.8% |
| 6138-7160 | 1023 | -0.2760 | +21.0% |
| 7161-8190 | 1030 | -0.2369 | +18.1% |

sign: 57.2% improved / 42.8% worsened / 3 tied; mean |delta| 0.6736 nats

| pos | token | base NLL | comp NLL | delta | context |
|---|---|---|---|---|---|
| 4381 | ` Miss` | 16.184 | 0.011 | -16.173 | ` the two third he danced with[ Miss]\nKing, and the two` |
| 3751 | ` partner` | 16.141 | 0.329 | -15.812 | `. Do let me ask my[ partner] to introduce you.”\n\n[` |
| 3966 | `party` | 14.685 | 0.842 | -13.843 | ` admired by the Netherfield\n[party]. Mr. Bingley had` |
| 4376 | ` two` | 17.500 | 5.433 | -12.068 | ` two next. Then, the[ two] third he danced with Miss\n` |
| 7151 | `me` | 3.896 | 15.344 | +11.448 | `, “does not offend _[me]_ so much as pride\n` |
| 5868 | `inv` | 8.234 | 19.129 | +10.894 | ` well bred, were not\n[inv]iting. In that respect his` |
| 6362 | `te` | 13.620 | 2.772 | -10.847 | `s had made him\ncour[te]ous.\n\nLady Lucas was` |
| 6891 | `“` | 15.906 | 5.174 | -10.733 | ` mistake?” said Jane.\n[“]I certainly saw Mr. Darcy` |
| 2766 | `was` | 4.776 | 15.254 | +10.478 | `. Mr. Bingley\n[was] obliged to be in town the` |
| 4404 | ` two` | 3.843 | 14.002 | +10.158 | ` Jane\nagain, and the[ two] sixth with Lizzy, and the` |
| 4436 | `\n` | 11.479 | 1.335 | -10.144 | `,” cried her husband impatiently,[\n]“he would not have danced` |
| 1687 | `\n` | 20.984 | 11.401 | -9.583 | ` I am not acquainted with him[\n]myself; how can you` |
| 5938 | ` been` | 1.387 | 10.919 | +9.532 | ` in his life; everybody had[ been] most kind and attentive to him` |
| 4583 | `agger` | 15.945 | 25.378 | +9.433 | ` spirit, and some\nex[agger]ation, the shocking rudeness` |
| 4706 | ` man` | 9.392 | 0.002 | -9.390 | `. I quite detest the[ man].”\n\n\n\n\n[Illustration]\n\n\n\n\n` |
| 7602 | ` _` | 11.198 | 20.284 | +9.086 | ` wish of being better acquainted with[ _]them_ was expressed towards the` |
| 263 | ` north` | 14.822 | 5.858 | -8.964 | ` man of large fortune from the[ north] of England; that he came` |
| 4607 | `that` | 14.881 | 6.044 | -8.837 | ` you,” she added, “[that] Lizzy does not lose much by` |
| 4312 | `up` | 10.489 | 1.655 | -8.835 | `ed to see him stand\n[up] with her; but, however` |
| 3502 | `friend` | 12.568 | 3.782 | -8.786 | ` few minutes to press his\n[friend] to join it.\n\n“` |
| 7412 | `s` | 19.888 | 11.436 | -8.452 | `, who came with his\n[s]isters, “I should not` |
| 5668 | ` tempted` | 12.242 | 20.606 | +8.364 | `age two years when he was[ tempted], by an accidental recommendation,` |
| 2754 | ` answer` | 13.566 | 5.747 | -7.820 | `\nhousekeeping, when an[ answer] arrived which deferred it all.` |
| 3815 | `“` | 7.733 | 0.010 | -7.724 | `, and coldly said,\n[“]She is tolerable: but not` |
| 6081 | ` should` | 10.111 | 2.433 | -7.678 | `, and one whom\nthey[ should] not object to know more of` |
| 4512 | `in` | 20.085 | 12.420 | -7.665 | ` charming women. I never\n[in] my life saw anything more elegant` |
| 5718 | ` praise` | 13.609 | 6.000 | -7.609 | ` what the owner said in its[ praise], and took it immediately.` |
| 567 | `\n` | 13.457 | 5.855 | -7.601 | ` my share of beauty, but[\n]I do not pretend to be` |
| 7756 | `was` | 14.001 | 6.588 | -7.413 | ` considered with pleasure that it\n[was] not likely to be discovered by` |
| 6836 | ` half` | 12.129 | 4.734 | -7.396 | `\nsat close to her for[ half] an hour without once opening his` |
| 7598 | ` being` | 14.254 | 6.910 | -7.344 | ` to,\na wish of[ being] better acquainted with _them_` |
| 7518 | `Illustration` | 7.606 | 0.338 | -7.268 | `\n\n[Illustration]\n\n\n\n\n[[Illustration]]\n\n\n\n\nCHAPTER VI.\n\n\n` |
| 5566 | ` Nether` | 9.397 | 2.146 | -7.251 | `\nremainder of his days at[ Nether]field, and leave the next` |
| 3984 | `inguished` | 9.258 | 2.064 | -7.194 | ` and she had been\ndist[inguished] by his sisters. Jane was` |
| 7266 | `lections` | 0.790 | 7.920 | +7.131 | ` the solidity of her\nref[lections], “is a very common` |
| 2842 | `sett` | 7.273 | 0.170 | -7.103 | ` to another, and never\n[sett]led at Netherfield as he` |
| 4330 | ` nobody` | 10.558 | 17.600 | +7.041 | ` her at all; indeed,[ nobody]\ncan, you know;` |
| 7118 | ` believe` | 15.564 | 8.525 | -7.039 | ` were you.”\n\n“I[ believe], ma’am, I` |
| 4510 | ` never` | 15.450 | 22.360 | +6.911 | ` sisters are charming women. I[ never]\nin my life saw anything` |
| 7496 | ` to` | 8.110 | 1.206 | -6.904 | ` she should not; she continued[ to] declare that she\nwould;` |
