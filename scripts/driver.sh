# StainID pipeline to segment the stains and analyze the results
python -m script.make_previews --input_dir data/TMA_core_exports --out_root results_stainID --max_per_type 6 --longest_side 2048
streamlit run script/calibrate_stain_streamlit.py
python -m script.segment_stain --input_dir data/TMA_core_exports --out_root results_stainID --types 212 213 214 215 --metadata data/brain_summary.csv
python -m script.analyze_results --out_root results_stainID --types 212 213 214 215 --metadata data/brain_summary.csv