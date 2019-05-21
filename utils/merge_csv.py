import pandas as pd


for color in ('black', 'white'):
    loss = []
    for size in ('single', 'bundle'):
        # acc = []
        # for data in ('accuracy', 'precision', 'recall'):
        #     for spliter in ('train', 'val'):
        #         path = f'{color}_{size}_runs/{data}_{spliter}-tag-{data}.csv'
        #         tb = pd.read_csv(path,  usecols=['Value'], )
        #         tb.rename(columns={'Value': f'{color}-{size}-{data}-{spliter}'}, inplace=True)
        #         acc.append(tb)
        # frame = pd.concat(acc, axis=1)
        # print(frame.shape)
        # frame.to_csv(f'{color}_{size}_runs/accuracies.csv')


        for data in ('loss',):
            for spliter in ('train', 'val'):
                path = f'{color}_{size}_runs/{data}_{spliter}-tag-{data}.csv'
                tb = pd.read_csv(path,  usecols=['Value'], )
                tb.rename(columns={'Value': f'{color}-{size}-{data}-{spliter}'}, inplace=True)
                loss.append(tb)
    f = pd.concat(loss, axis=1)
    print(f.shape)
    f.to_csv(f'{color}_losses.csv')