# Evaluation Summary (Markdown)

Generated from `train_result\2025-11-21\run_001`

| ablation_key       | ablation_value   | name                             | train_mode   | adversarial   | flow_preprocessing   |   test_ari |   test_nmi | model_path                                                                                                |
|:-------------------|:-----------------|:---------------------------------|:-------------|:--------------|:---------------------|-----------:|-----------:|:----------------------------------------------------------------------------------------------------------|
| adversarial        | kl               | adversarial_kl_best              | gated        | kl            | centered             |   0.47378  |   0.59862  | train_result\2025-11-21\run_001\ablation\adversarial\kl\adversarial_kl_best.pth                           |
| train_mode         | gated            | train_mode_gated_best            | gated        | kl            | centered             |   0.444651 |   0.577731 | train_result\2025-11-21\run_001\ablation\train_mode\gated\train_mode_gated_best.pth                       |
| flow_preprocessing | centered         | flow_preprocessing_centered_best | gated        | kl            | centered             |   0.438658 |   0.566935 | train_result\2025-11-21\run_001\ablation\flow_preprocessing\centered\flow_preprocessing_centered_best.pth |
| adversarial        | off              | adversarial_off_best             | gated        | off           | centered             |   0.42319  |   0.559985 | train_result\2025-11-21\run_001\ablation\adversarial\off\adversarial_off_best.pth                         |
| flow_preprocessing | normal           | flow_preprocessing_normal_best   | gated        | kl            | normal               |   0.455624 |   0.557942 | train_result\2025-11-21\run_001\ablation\flow_preprocessing\normal\flow_preprocessing_normal_best.pth     |
| adversarial        | gan              | adversarial_gan_best             | gated        | gan           | centered             |   0.406981 |   0.531823 | train_result\2025-11-21\run_001\ablation\adversarial\gan\adversarial_gan_best.pth                         |
| train_mode         | flow             | train_mode_flow_best             | flow         | kl            | centered             |   0.322391 |   0.468581 | train_result\2025-11-21\run_001\ablation\train_mode\flow\train_mode_flow_best.pth                         |
| train_mode         | mae              | train_mode_mae_best              | mae          | kl            | centered             |   0.187488 |   0.359988 | train_result\2025-11-21\run_001\ablation\train_mode\mae\train_mode_mae_best.pth                           |
