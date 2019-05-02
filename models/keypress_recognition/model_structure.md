# Keypress Recognition Network Structure

The previous localization network shall output a 884\*106 image. We will acquire the configs of `key_seperation` algorithm to calculate the dimentions of the parameters. According to the current settings of `key_seperation` algorithm, each white key will be of size 21\*106, and black key 12\*106.

## White key

| Input Size  | Network Layer  |
|---|---|
| 21\*106\*1 | 3\*3 Conv, 16 filters with padding |
| 21\*106\*16 | 3\*3 Conv, 16 filters with padding |
| 21\*106\*16 | 2\*2 Max Pool |
| 10\*53\*16 | 3\*3 Conv, 32 filters with padding |
| 10\*53\*32 | 3\*3 Conv, 32 filters with padding |
| 10\*53\*32 | 2\*2 Max Pool |
| 5\*26\*32 | FC 4160-> 512 |
| 512 | FC 512-> 1 |

## Black key

| Input Size  | Network Layer  |
|---|---|
| 12\*106\*1 | 3\*3 Conv, 16 filters with padding |
| 12\*106\*16 | 3\*3 Conv, 16 filters with padding |
| 12\*106\*16 | 2\*2 Max Pool |
| 6\*53\*16 | 3\*3 Conv, 32 filters with padding |
| 6\*53\*32 | 3\*3 Conv, 32 filters with padding |
| 6\*53\*32 | 2\*2 Max Pool |
| 3\*26\*32 | FC 2496-> 512 |
| 512 | FC 512-> 1 |

## Dimension reference

[this post](https://music.stackexchange.com/a/53872)