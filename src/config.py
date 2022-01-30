import os
import os.path as osp

SEED = 42
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# paths
SS_PATH = osp.join(ROOT_DIR,"data","raw","sample_submission.csv")
TEST_DATA_PROCESSED00 = osp.join(ROOT_DIR,"data","processed","test_feat_eng_00.csv")
TEST_DATA_PROCESSED01 = osp.join(ROOT_DIR,"data","processed","test_feat_eng_01.csv")


# cat features
CAT_FEATURES = ['country','store','product']

if __name__ == '__main__':
    print("Root directory:",ROOT_DIR)