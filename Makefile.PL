use inc::Module::Install;

name          'ToyBox-PassiveAggressive';
all_from      'lib/ToyBox/PassiveAggressive.pm';
license       'MIT';

test_requires 'Test::More';

WriteAll;
