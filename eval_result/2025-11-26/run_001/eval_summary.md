# Evaluation Summary (Markdown)

Generated from `train_result\2025-11-26\run_001`

| ablation_key       | ablation_value   | name                             | train_mode   | adversarial   | flow_preprocessing   |   test_ari |   test_nmi | model_path                                                                                                |
|:-------------------|:-----------------|:---------------------------------|:-------------|:--------------|:---------------------|-----------:|-----------:|:----------------------------------------------------------------------------------------------------------|
| train_mode         | mae              | train_mode_mae_best              | mae          | kl            | normal               |   0.205557 |   0.291401 | train_result\2025-11-26\run_001\ablation\train_mode\mae\train_mode_mae_best.pth                           |
| adversarial        | gan              | adversarial_gan_best             | gated        | gan           | normal               | nan        | nan        | train_result\2025-11-26\run_001\ablation\adversarial\gan\adversarial_gan_best.pth                         |
| adversarial        | kl               | adversarial_kl_best              | gated        | kl            | normal               | nan        | nan        | train_result\2025-11-26\run_001\ablation\adversarial\kl\adversarial_kl_best.pth                           |
| adversarial        | off              | adversarial_off_best             | gated        | off           | normal               | nan        | nan        | train_result\2025-11-26\run_001\ablation\adversarial\off\adversarial_off_best.pth                         |
| flow_preprocessing | centered         | flow_preprocessing_centered_best | gated        | kl            | centered             | nan        | nan        | train_result\2025-11-26\run_001\ablation\flow_preprocessing\centered\flow_preprocessing_centered_best.pth |
| flow_preprocessing | normal           | flow_preprocessing_normal_best   | gated        | kl            | normal               | nan        | nan        | train_result\2025-11-26\run_001\ablation\flow_preprocessing\normal\flow_preprocessing_normal_best.pth     |
| train_mode         | flow             | train_mode_flow_best             | flow         | kl            | normal               | nan        | nan        | train_result\2025-11-26\run_001\ablation\train_mode\flow\train_mode_flow_best.pth                         |
| train_mode         | gated            | train_mode_gated_best            | gated        | kl            | normal               | nan        | nan        | train_result\2025-11-26\run_001\ablation\train_mode\gated\train_mode_gated_best.pth                       |
