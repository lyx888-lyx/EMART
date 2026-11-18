import unittest
import pandas as pd
from train_split_gen.v11_audit_and_prepare import identity, make_contexts


class ToyTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [0]*(len(text.split())+2)


class TestV11(unittest.TestCase):
    def test_speaker_is_suffix_not_recording_prefix(self):
        self.assertEqual(identity("Ses01F_impro01_F000"),
                         ("Ses01F_impro01","F"))
        self.assertEqual(identity("Ses01F_impro01_M000"),
                         ("Ses01F_impro01","M"))
        self.assertEqual(identity("Ses01F_impro01_F001"),
                         ("Ses01F_impro01","F"))
        with self.assertRaises(ValueError):
            identity("F000")

    def test_correct_role_and_order(self):
        df=pd.DataFrame([
            ["d",3.0,"Ses01F_impro01_F001","later","ang"],
            ["d",1.0,"Ses01F_impro01_F000","earlier","neu"],
            ["d",2.0,"Ses01F_impro01_M000","response","sad"],
        ],columns=["Dialogue_ID","Start_Time","Speaker","Utterance","Emotion"])
        x=make_contexts(df,ToyTokenizer())
        r=x["Ses01F_impro01_F001"]
        self.assertEqual(r["legacy"],"<OTHER> earlier </s></s> <OTHER> response </s></s> [Current] <SELF> later")
        self.assertEqual(r["role_fixed"],"<SELF> earlier </s></s> <OTHER> response </s></s> [Current] <SELF> later")
        self.assertEqual(r["historical_self"],1)
        self.assertEqual(x["Ses01F_impro01_F000"]["role_fixed"],"[Current] <SELF> earlier")
        self.assertNotIn("later",x["Ses01F_impro01_M000"]["role_fixed"])

    def test_dialogues_do_not_mix(self):
        df=pd.DataFrame([
            ["d",1.0,"Ses01F_impro01_F000","a","neu"],
            ["d",2.0,"Ses01F_impro02_M000","b","sad"],
        ],columns=["Dialogue_ID","Start_Time","Speaker","Utterance","Emotion"])
        with self.assertRaises(ValueError):
            make_contexts(df,ToyTokenizer())


if __name__=="__main__":
    unittest.main()
