def config():
    ref = 11.6
    ref2 = ref * 2
    ref3 = ref * 3
    ref4 = ref * 4
    return {
        # image size -> acc
        's720':{
            'sigma_points': [ref3, ref3, ref3],
            'sigma_links': [ref3, ref3, ref3],
            'img_size': 720,
        },
        's360':{
            'sigma_points': [ref3, ref3, ref3],
            'sigma_links': [ref3, ref3, ref3],
            'img_size': 360,
        },
        's256':{
            'sigma_points': [ref3, ref3, ref3],
            'sigma_links': [ref3, ref3, ref3],
            'img_size': 256,
        },
        's128':{
            'sigma_points': [ref3, ref3, ref3],
            'sigma_links': [ref3, ref3, ref3],
            'img_size': 128,
        },
        's64':{
            'sigma_points': [ref3, ref3, ref3],
            'sigma_links': [ref3, ref3, ref3],
            'img_size': 64,
        },

        # sigma size -> acc
        # update img_size after get result of img
        '1x': {
            'sigma_points': [ref *1, ref*1, ref*1],
            'sigma_links': [ref *1, ref*1, ref*1],
            'img_size': 720,
        },
        '2x': {
            'sigma_points': [ref *2, ref*2, ref*2],
            'sigma_links': [ref *2, ref*2, ref*2],
            'img_size': 720,
        },
        '3x': {
            'sigma_points': [ref *3, ref*3, ref*3],
            'sigma_links': [ref *3, ref*3, ref*3],
            'img_size': 720,
        },
        '4x': {
            'sigma_points': [ref *4, ref*4, ref*4],
            'sigma_links': [ref *4, ref*4, ref*4],
            'img_size': 720,
        },
        '5x': {
            'sigma_points': [ref *5, ref*5, ref*5],
            'sigma_links': [ref *5, ref*5, ref*5],
            'img_size': 720,
        },
        '6x': {
            'sigma_points': [ref *6, ref*6, ref*6],
            'sigma_links': [ref *6, ref*6, ref*6],
            'img_size': 720,
        },
        '8x': {
            'sigma_points': [ref *8, ref*8, ref*8],
            'sigma_links': [ref *8, ref*8, ref*8],
            'img_size': 720,
        },
        '20x': { # 20x so bad result
            'sigma_points': [ref *20, ref*20, ref*20],
            'sigma_links': [ref *20, ref*20, ref*20],
            'img_size': 720,
        },
        '9x': {
            'sigma_points': [ref *9, ref*9, ref*9],
            'sigma_links': [ref *9, ref*9, ref*9],
            'img_size': 720,
        },
        '10x': {
            'sigma_points': [ref *10, ref*10, ref*10],
            'sigma_links': [ref *10, ref*10, ref*10],
            'img_size': 720,
        },
        '12x': {
            'sigma_points': [ref *12, ref*12, ref*12],
            'sigma_links': [ref *12, ref*12, ref*12],
            'img_size': 720,

        },
        '1y': {
            'sigma_points': [ref *1, ref*1, ref*1],
            'sigma_links': [ref *1, ref*1, ref*1],
            'img_size': 360,
        },
        '4y': {
            'sigma_points': [ref *4, ref*4, ref*4],
            'sigma_links': [ref *4, ref*4, ref*4],
            'img_size': 360,
        },
        '7y': {
            'sigma_points': [ref *7, ref*7, ref*7],
            'sigma_links': [ref *7, ref*7, ref*7],
            'img_size': 360,
        },
        '10y': {
            'sigma_points': [ref *10, ref*10, ref*10],
            'sigma_links': [ref *10, ref*10, ref*10],
            'img_size': 360,
        },
        '13y': {
            'sigma_points': [ref *13, ref*13, ref*13],
            'sigma_links': [ref *13, ref*13, ref*13],
            'img_size': 360,
        },

        # curriculum training -> acc
        # update img_size after get result of img

        # ++   + ref2
        # ref4 - --
        '8p': {
            'sigma_points': [ref4, ref4*.9, ref4*.8],
            'sigma_links': [ref4, ref4*.9, ref4*.8],
            'img_size': 360,
        },
        '6p': {
            'sigma_points': [ref4, ref4*.8, ref4*.6],
            'sigma_links': [ref4, ref4*.8, ref4*.6],
            'img_size': 360,
        },
        '4p': {
            'sigma_points': [ref4, ref4*.7, ref4*.4],
            'sigma_links': [ref4, ref4*.7, ref4*.4],
            'img_size': 360,
        },
        '2p': {
            'sigma_points': [ref4, ref4*.6, ref4*.2],
            'sigma_links': [ref4, ref4*.6, ref4*.2],
            'img_size': 360,
        },
        '1p': {
            'sigma_points': [ref4, ref4*.5, ref4*.1],
            'sigma_links': [ref4, ref4*.5, ref4*.1],
            'img_size': 360,
        },

    }
