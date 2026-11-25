### Bonus A – LLM Evaluation

The LLM predictions were compared with 20 manually labelled comments provided by WEFRA LIFE.

The final evaluation achieved **100% topic accuracy, 90% sentiment accuracy, and 100% precision and recall for side-effect detection**. All three human-labelled potential side effects were correctly identified, with no false negatives.

Two sentiment errors occurred in identical cryptocurrency spam comments (CM000217 and CM002180). Although the LLM correctly identified both as spam, it classified their sentiment as negative instead of neutral. This suggests the model may confuse undesirable promotional content with negative emotional sentiment.

Prompt refinements improved topic and sentiment classification while maintaining side-effect recall. One text mismatch between the labelled dataset and the database was also identified and documented.

These results are encouraging but limited by the small labelled sample and the fact that the same examples informed prompt improvements. Further evaluation on unseen comments is needed to assess generalization.