# @function: feature importance computation
# @author linna Fan
# input: the trained model
# output: importance of each feature
from plot_util import plot_importance_barh

class FeatureImportance(object):
    # model is the trained model, col_names are feature names, common_methods are list of importance evaluation methods
    def __init__(self,model,col_names,common_methods,saved_path,train_x,train_y,test_x,test_y):
        self.model = model
        self.col_names = col_names[3:]
        self.common_methods = common_methods
        self.saved_path = saved_path
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
    # get importance through common methods
    def get_importance_common_methods(self):
        for common_method in self.common_methods:
            if common_method == 'default':
                plot_importance_barh(self.model.feature_importances_,self.saved_path + '/importance_{}.png'.format(common_method),self.col_names)
                pass
            elif common_method == 'permutation':
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(self.model, self.test_x, self.test_y)
                # from rfpimp import permutation_importances
                # from sklearn.metrics import r2_score
                # def r2(rf, X_train, y_train):
                #     return r2_score(y_train, rf.predict(X_train))
                # perm_imp_rfpimp = permutation_importances(self.model, self.train_x, self.train_y, r2)
                plot_importance_barh(perm_importance.importances_mean,self.saved_path + '/importance_{}.png'.format(common_method),self.col_names)


