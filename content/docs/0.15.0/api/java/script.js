/*========================================================================
 *  Licensed Materials - Property of IBM
 *  "Restricted Materials of IBM"
 *
 *  IBM SDK, Java(tm) Technology Edition, v8
 *  (C) Copyright IBM Corp. 2000, 2014. All Rights Reserved
 *
 *  US Government Users Restricted Rights - Use, duplication or disclosure
 *  restricted by GSA ADP Schedule Contract with IBM Corp.
 *========================================================================
 */
function show(type)
{
    count = 0;
    for (var key in methods) {
        var row = document.getElementById(key);
        if ((methods[key] &  type) != 0) {
            row.style.display = '';
            row.className = (count++ % 2) ? rowColor : altColor;
        }
        else
            row.style.display = 'none';
    }
    updateTabs(type);
}

function updateTabs(type)
{
    for (var value in tabs) {
        var sNode = document.getElementById(tabs[value][0]);
        var spanNode = sNode.firstChild;
        if (value == type) {
            sNode.className = activeTableTab;
            spanNode.innerHTML = tabs[value][1];
        }
        else {
            sNode.className = tableTab;
            spanNode.innerHTML = "<a href=\"javascript:show("+ value + ");\">" + tabs[value][1] + "</a>";
        }
    }
}
