import functools
import types

import tensorflow as tf


def operation(scope_or_func=None, **kwargs):
    '''
    Decorator which wraps a function which builds a tensorflow operation.

    This will make the function a property of the class with a cached
    operation which was built by the function.

    Arguments:
      function: Function which builds a tensorflow operation
      scope: Scope to wrap with tf.variable_scope
    '''
    has_scope = not isinstance(scope_or_func, types.FunctionType)
    if has_scope:
        def wraped(function):
            attribute = '__cache_%s' % function.__name__
            scope = scope_or_func if scope_or_func is not None else function.__name__

            @property
            @functools.wraps(function)
            def decorator(self):
                if not hasattr(self, attribute):
                    with tf.variable_scope(scope, **kwargs):
                        setattr(self, attribute, function(self))
                    setattr(getattr(self, attribute), 'scope', scope)
                    setattr(getattr(self, attribute), 'operation', True)
                return getattr(self, attribute)
            return decorator
        return wraped

    attribute = '__cache_%s' % scope_or_func.__name__
    scope = scope_or_func.__name__

    @property
    @functools.wraps(scope_or_func)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(scope):
                setattr(self, attribute, scope_or_func(self))
                setattr(getattr(self, attribute), 'scope', scope)
                setattr(getattr(self, attribute), 'operation', True)
        return getattr(self, attribute)
    return decorator
