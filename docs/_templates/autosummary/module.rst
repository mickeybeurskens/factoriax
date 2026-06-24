{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :no-imported-members:

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block modules %}
   {%- if modules %}
   .. rubric:: Modules

   .. autosummary::
      :toctree:
      :recursive:
   {% for item in modules %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}
