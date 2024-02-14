#import mlflow
#from kedro.framework.hooks import hook_impl

#class MLflowHook:
  #  @hook_impl
  #  def before_node_run(self, node, catalog, inputs):
      #  mlflow.start_run(run_name=node.name)

#    @hook_impl
   # def after_node_run(self, node, catalog, inputs, outputs):
    #    mlflow.end_run()
