import pandas as pd

import numpy as np

importances = rf_model.feature_importances_

feature_names = vectorizer.get_feature_names_out()

indices = np.argsort(importances)[-15:]

plt.barh(range(len(indices)), importances[indices], align='center')

plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

plt.xlabel('Relative Importance')

plt.title('Top 15 Important Features (Words)')

plt.show()
