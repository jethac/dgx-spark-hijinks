## strata_nvfp4_vs_bf16_c2

baseline `claude_bf16_vosplit_c2_ctx8191` vs comparator `claude_nvfp4_c2_ctx8191` — 8190 tokens, mean delta **+0.2526** nats (negative = quantized better)

| baseline-NLL band | count | mean delta | share of total |
|---|---|---|---|
| [0,0.5) | 2426 | +0.2432 | +28.5% |
| [0.5,1) | 316 | +0.8630 | +13.2% |
| [1,2) | 422 | +0.6654 | +13.6% |
| [2,4) | 745 | +0.6491 | +23.4% |
| [4,8) | 1474 | +0.5449 | +38.8% |
| >=8 | 2807 | -0.1289 | -17.5% |

| positions | count | mean delta | share of total |
|---|---|---|---|
| 0-1022 | 1022 | +0.0389 | +1.9% |
| 1023-2045 | 1023 | +0.0731 | +3.6% |
| 2046-3068 | 1023 | +0.0919 | +4.5% |
| 3069-4091 | 1023 | +0.3504 | +17.3% |
| 4092-5114 | 1023 | +0.2307 | +11.4% |
| 5115-6137 | 1023 | +0.4954 | +24.5% |
| 6138-7160 | 1023 | +0.4037 | +20.0% |
| 7161-8190 | 1030 | +0.3357 | +16.7% |

sign: 43.5% improved / 56.4% worsened / 3 tied; mean |delta| 1.1047 nats

| pos | token | base NLL | comp NLL | delta | context |
|---|---|---|---|---|---|
| 5758 | `iness` | 0.000 | 18.529 | +18.529 | ` to Darcy by the\neas[iness], openness, and ductility of` |
| 3246 | `friend` | 2.683 | 20.075 | +17.392 | ` to be compared with his\n[friend].\n\nMr. Bingley` |
| 6865 | ` George` | 0.047 | 16.971 | +16.925 | ` 1894 by[ George] Allen._]]\n\n“Are` |
| 4740 | ` Mr` | 0.043 | 14.026 | +13.983 | ` cautious in\nher praise of[ Mr]. Bingley before, expressed` |
| 250 | `field` | 0.000 | 13.268 | +13.268 | ` Mrs. Long says that Nether[field] is taken\nby a young` |
| 4376 | ` two` | 17.500 | 4.738 | -12.762 | ` two next. Then, the[ two] third he danced with Miss\n` |
| 5276 | ` temper` | 2.549 | 15.169 | +12.620 | ` and less pliancy of[ temper] than her sister, and\n` |
| 4547 | ` Bennet` | 0.009 | 12.550 | +12.541 | ` was interrupted again. Mr.[ Bennet] protested against any\ndescription of` |
| 7521 | `CHAPTER` | 0.053 | 12.001 | +11.948 | `]\n\n\n\n\n[Illustration]\n\n\n\n\n[CHAPTER] VI.\n\n\n[Illustration]` |
| 7266 | `lections` | 0.790 | 12.425 | +11.635 | ` the solidity of her\nref[lections], “is a very common` |
| 4089 | ` lived` | 14.209 | 2.638 | -11.572 | `n, the village where they[ lived], and of which they\n` |
| 3497 | ` minutes` | 3.499 | 14.332 | +10.833 | ` from the dance for a few[ minutes] to press his\nfriend to` |
| 3295 | `him` | 18.743 | 7.946 | -10.797 | ` and talked of giving one\n[him]self at Netherfield. Such` |
| 6216 | ` been` | 0.351 | 11.128 | +10.777 | `distinction had, perhaps,[ been] felt too strongly. It had` |
| 4079 | `its` | 10.805 | 21.386 | +10.581 | ` therefore, in good\nspir[its] to Longbourn, the` |
| 7300 | `nature` | 6.583 | 17.164 | +10.581 | ` common indeed; that human\n[nature] is particularly prone to it,` |
| 4134 | ` to` | 0.072 | 10.480 | +10.408 | `\ngood deal of curiosity as[ to] the event of an evening which` |
| 5382 | `\n` | 0.482 | 10.702 | +10.220 | ` thousand pounds; were in the[\n]habit of spending more than they` |
| 243 | `,` | 13.376 | 3.189 | -10.187 | ` my dear, you must know[,] Mrs. Long says that Nether` |
| 2861 | `little` | 3.075 | 13.191 | +10.117 | ` quieted her fears a\n[little] by starting the idea of his` |
| 7814 | ` Miss` | 4.049 | 14.130 | +10.081 | ` mentioned this to her friend,[ Miss]\nLucas.\n\n“It` |
| 3835 | `hum` | 10.368 | 20.398 | +10.030 | ` and I am in no\n[hum]our at present to give consequence` |
| 4584 | `ation` | 17.483 | 7.483 | -10.000 | `, and some\nexagger[ation], the shocking rudeness of` |
| 3367 | ` own` | 5.712 | 15.671 | +9.958 | ` speaking occasionally to one of his[ own] party.\nHis character was` |
| 5447 | ` their` | 14.938 | 5.292 | -9.646 | ` their\nmemories than that[ their] brother’s fortune and their` |
| 7308 | ` that` | 0.157 | 9.658 | +9.501 | ` particularly prone to it, and[ that] there are very few of us` |
| 192 | ` I` | 10.771 | 1.287 | -9.485 | ` want to tell me, and[ I] have no objection to hearing it` |
| 3599 | ` room` | 0.004 | 9.443 | +9.439 | ` not\nanother woman in the[ room] whom it would not be a` |
| 236 | `,` | 7.174 | 16.533 | +9.359 | ` invitation enough.\n\n“Why[,] my dear, you must know` |
| 80 | ` other` | 11.603 | 2.250 | -9.353 | `\nproperty of some one or[ other] of their daughters.\n\n“` |
| 5068 | ` is` | 0.024 | 9.134 | +9.110 | ` know you do: and it[ is] _that_ which makes the` |
| 5019 | ` your` | 0.005 | 9.084 | +9.079 | ` are good and agreeable\nin[ your] eyes. I never heard you` |
| 6578 | `about` | 0.708 | 9.750 | +9.043 | ` hardly know what--something\n[about] Mr. Robinson.”\n\n“` |
| 4102 | ` found` | 8.386 | 17.415 | +9.028 | `were the principal inhabitants. They[ found] Mr. Bennet still up.` |
| 6358 | ` made` | 14.285 | 5.313 | -8.972 | ` St. James’s had[ made] him\ncourteous.` |
| 5643 | `his` | 2.641 | 11.566 | +8.925 | `, less disposed to consider\n[his] house as her home when it` |
| 5552 | ` temper` | 4.393 | 13.309 | +8.916 | ` knew the easiness of his[ temper], whether he might not spend` |
| 2743 | ` to` | 0.001 | 8.893 | +8.892 | ` Bennet planned the courses that were[ to] do credit to her\nhouse` |
| 7118 | ` believe` | 15.564 | 6.713 | -8.851 | ` were you.”\n\n“I[ believe], ma’am, I` |
| 6439 | `\n` | 5.213 | 14.019 | +8.806 | ` should meet to talk over a[\n]ball was absolutely necessary; and` |
